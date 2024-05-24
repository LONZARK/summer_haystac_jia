## h3_convert code

This Docker container can be used to build `h3_convert`, which is a binary that can calculate the [h3 index](https://www.uber.com/blog/h3/) for a set of points in a csv file.

### Building `h3_convert`

The included Dockerfile creates a Docker container which can be used to quickly build `h3_convert` for Ubuntu 20.04. First modify `h3.yaml` to point to the *src* directory on your disk. Then build and start the container.

```sh
docker build . -t h3
docker-compose -f h3.yaml up -d
```

After the container is running you can run the build.

```sh
docker exec -ti /bin/bash
cd /home/dev
mkdir build
cd build
cmake ..
make -j
```

After the build is complete you can copy the `h3_convert` binary from `./src/build/bin` and run it on your local machine (assuming it is Ubuntu 20.04).


### Running `h3_convert`

```sh
h3_convert input_file.csv output_file.csv
```

**NOTE** The input csv file must have a `Latitude` column and a `Longitude` column. The resulting output file will have 16 new columns added with the format `res_{0-15}` indicating the h3 index of the point at various resolution levels. All of the original data from the input file will also be included.