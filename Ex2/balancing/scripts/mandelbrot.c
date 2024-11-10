#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

unsigned char mandelbrot(double real, double imag, int max_iter) {
    double z_real = real;
    double z_imag = imag;
    for (int n = 0; n < max_iter; n++) {
        double r2 = z_real * z_real;
        double i2 = z_imag * z_imag;
        if (r2 + i2 > 4.0) return n;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = r2 - i2 + real;
    }
    return max_iter;
}

int main(int argc, char *argv[]) {
    int mpi_provided_thread_level;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_provided_thread_level);
    if (mpi_provided_thread_level < MPI_THREAD_FUNNELED) {
        printf("The threading support level is lesser than that demanded\n");
        MPI_Finalize();
        exit(1);
    }

    double global_start_time = MPI_Wtime();

    if (argc == 9) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        x_left = atof(argv[3]);
        y_lower = atof(argv[4]);
        x_right = atof(argv[5]);
        y_upper = atof(argv[6]);
        max_iterations = atoi(argv[7]);
        num_threads = atoi(argv[8]);
        omp_set_num_threads(num_threads);
    }

    int world_size, world_rank, num_threads;
    const int low_res_factor = 10;  // Fattore di riduzione per la griglia di complessità
    int low_res_width = width / low_res_factor;
    int low_res_height = height / low_res_factor;
    int *complexity_map = NULL;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Calcolo della griglia di complessità (solo il processo root)
    if (world_rank == 0) {
        complexity_map = malloc(low_res_width * low_res_height * sizeof(int));
        for (int j = 0; j < low_res_height; j++) {
            for (int i = 0; i < low_res_width; i++) {
                double x = x_left + i * (x_right - x_left) / low_res_width;
                double y = y_lower + j * (y_upper - y_lower) / low_res_height;
                complexity_map[j * low_res_width + i] = mandelbrot(x, y, max_iterations);
            }
        }
    }

    // Broadcast della mappa di complessità a tutti i processi
    if (world_rank == 0) {
        MPI_Bcast(complexity_map, low_res_width * low_res_height, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        complexity_map = malloc(low_res_width * low_res_height * sizeof(int));
        MPI_Bcast(complexity_map, low_res_width * low_res_height, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Divisione del carico in base alla complessità stimata
    int rows_per_process = height / world_size;
    int remainder_rows = height % world_size;
    int start_row = world_rank * rows_per_process + (world_rank < remainder_rows ? world_rank : remainder_rows);
    int end_row = start_row + rows_per_process + (world_rank < remainder_rows ? 1 : 0);

    unsigned char* part_buffer = (unsigned char*)malloc(width * (end_row - start_row) * sizeof(unsigned char));

#pragma omp parallel for schedule(dynamic)
    for (int j = start_row; j < end_row; j++) {
        for (int i = 0; i < width; i++) {
            double x = x_left + i * (x_right - x_left) / width;
            double y = y_lower + j * (y_upper - y_lower) / height;
            int index = (j - start_row) * width + i;
            part_buffer[index] = mandelbrot(x, y, max_iterations);
        }
    }

    // Preparazione per MPI_Gatherv
    int *recvcounts = NULL;
    int *displs = NULL;
    unsigned char *image_buffer = NULL;
    if (world_rank == 0) {
        recvcounts = malloc(world_size * sizeof(int));
        displs = malloc(world_size * sizeof(int));
        int displacement = 0;
        for (int i = 0; i < world_size; i++) {
            recvcounts[i] = width * (rows_per_process + (i < remainder_rows ? 1 : 0));
            displs[i] = displacement;
            displacement += recvcounts[i];
        }
        image_buffer = (unsigned char*)malloc(displacement * sizeof(unsigned char));
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Sincronizzazione prima della raccolta
    MPI_Gatherv(part_buffer, width * (end_row - start_row), MPI_UNSIGNED_CHAR,
                image_buffer, recvcounts, displs, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    free(part_buffer);

    double global_end_time = MPI_Wtime();
    if(world_rank == 0) {
       printf("%f", global_end_time - global_start_time);
    }

    // Salvataggio dell'immagine (solo nel processo root)
    if (world_rank == 0) {
        FILE *file = fopen("image.pgm", "w");
        fprintf(file, "P5\n%d %d\n255\n", width, height);
        fwrite(image_buffer, sizeof(unsigned char), width * height, file);
        fclose(file);
        free(image_buffer);
        free(recvcounts);
        free(displs);
    }

    free(complexity_map);
    MPI_Finalize();
    return 0;
}

