#!/bin/bash

mkdir raster_data_H
python -c "from make_files import Photothermal_Files as pt_files; pt_files().prepare_initial_calculations()"

