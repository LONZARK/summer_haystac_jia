/*
 * Copyright 2017, 2020-2021 Uber Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Example program that converts coordinates to an H3 index (hexagon),
 * and then finds the vertices and center coordinates of the index.
 */

#include <h3/h3api.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LEN 512

int get_index(char *line, const char *label)
{
    int current_index = 0;
    char * entry;
    char *line_copy = (char *)malloc(strlen(line) + 1);
    strcpy(line_copy,line);
    entry=strtok(line_copy,",\n");
    while (entry!=NULL)
    {
        if (strcmp(entry, label)==0)
        {
            free(line_copy);
            return current_index;
        }
        current_index = current_index + 1;
        entry=strtok(NULL,",\n");
    }
    free(line_copy);
    return -1;
}

double get_value(char * line, int index)
{
    int current_index = 0;
    char * entry;
    char *line_copy = (char *)malloc(strlen(line) + 1);
    strcpy(line_copy,line);
    entry=strtok(line_copy,",\n");
    double output;
    while (entry!=NULL)
    {
        if (current_index==index)
        {
            free(line_copy);
            return strtod(entry, NULL);
        }
        current_index = current_index + 1;
        entry=strtok(NULL,",\n");
    }
    free(line_copy);
}

int main(int argc, char *argv[]) {
    // Read CSV and add H3 Indexes for each point
    
    // Open Input and Output Files
    char *input_file = argv[1];
    char *output_file = argv[2];
    FILE *fptr;
    FILE *ofptr;
    fptr = fopen(input_file, "r");
    ofptr = fopen(output_file, "w");

    // Line Buffer
    char line[MAX_LEN];

    // Find latitude and longitude in the file
    fgets(line, MAX_LEN, fptr);
    int lat_index = get_index(line, "Latitude");
    if (lat_index<0)
    {
        puts("Can't find Latitude column in file\n");
        exit(-1);
    }
    int lon_index = get_index(line, "Longitude");
    if (lon_index<0)
    {
        puts("Can't find Longitude column in file\n");
        exit(-1);
    }
    char * header = strtok(line,"\n");
    fprintf(ofptr,"%s",header);
    for(int i=0;i<16;i++)
    {
        fprintf(ofptr,",res_%d",i);
    }
    fprintf(ofptr,"\n");
    printf("Latitude Column Index %d\n", lat_index);
    printf("Longitude Column Index %d\n", lon_index);

    while (fgets(line, MAX_LEN, fptr) != NULL)
    {
        float latitude = get_value(line, lat_index);
        float longitude = get_value(line, lon_index);
        LatLng location;
        location.lat = degsToRads(latitude);
        location.lng = degsToRads(longitude);
        int resolution = 1;
        H3Index indexed;
        header = strtok(line,"\n");
        fprintf(ofptr,"%s",header);
        for(resolution=0;resolution<16;resolution++)
        {
            if (latLngToCell(&location, resolution, &indexed) != E_SUCCESS) {
                printf("Failed\n");
                return 1;
            }
            
            fprintf(ofptr,",%"PRIx64,indexed);
            // printf("The index is: %" PRIx64 " at resolution %d\n", indexed[resolution], resolution);
        }
        fprintf(ofptr,"\n");
    }
    fclose(ofptr);
    puts("File Conversion Complete\n");
}