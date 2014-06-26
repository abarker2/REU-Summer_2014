#include "output.h"

void output(char const* filename, float * grid, int x, int y, int z) {
	FILE * file;
	file = fopen(filename, "w+");
	
	for(int k=0; k < z; k++) {
		for(int j=0; j < y; j++) {
			for(int i=0; i < x; i++) {
				fprintf(file, "%f\n", grid[i + j*y + k*y*y]);
			}
			fprintf(file, "ROW\n");
		}
		fprintf(file, "BREAK\n");
	}
	fprintf(file, "EOF\n");
}
