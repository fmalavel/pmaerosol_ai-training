###############################################################################
# guess_coord_names : find coordinates' names
###############################################################################

def guess_coord_names(cube, axes):
    """
    Guess the name of the coordinate corresponding to the required axes

    :param cube: iris Cube
    :param axes: List of axes, eg 'X','Y','Z','T'

    :returns: List of coordinate names corresponding to these axes.
              If an axes not found, then value in list is None.
              Will try to return dimension coordinates if possible.
    """

    import iris

    coord_names = [None] * len(axes)
    for coord in cube.coords():
        axis = iris.util.guess_coord_axis(coord)
        for i, ax in enumerate(axes):
            if axis == ax:
                if coord_names[i] is None:
                    coord_names[i] = coord.name()

    return coord_names


###############################################################################
# extract_cube_area: subsample part of a cube's domain 
# for cubes on rotated coordinates
###############################################################################

def extract_cube_area(cube=None, 
                      MINLON=None,
                      MAXLON=None,
                      MINLAT=None,
                      MAXLAT=None):
    """
    Function that extract a region from a cube that is not on 
    a regular latlon grid.

    :cube:      an iris cube
    :MINLON:    Minimum longitude to be used constraining the cube
    :MAXLON:    Maximum longitude to be used constraining the cube
    :MINLAT:    Minimum latitude to be used constraining the cube
    :MAXLAT:    Minimum latitude to be used constraining the cube
    """

    import iris
    from iris.analysis.cartography import rotate_pole, unrotate_pole
    import numpy as np

    if MINLON is None:
        MINLON = -11.5
    if MAXLON is None:
        MAXLON = 5
    if MINLAT is None:
        MINLAT = 48.5
    if MAXLAT is None:
        MAXLAT = 61

    if cube is None:
        raise ValueError(f"No cube has been specified")
    
    if not isinstance(cube, iris.cube.Cube):
        raise ValueError(f"{cube} is not an Iris cube")

    ycoord, xcoord = None, None
    ycoord, xcoord  = guess_coord_names(cube, ["Y", "X"])
    #print(cube.coord(xcoord))
    #print(cube.coord(ycoord))

    if None in [xcoord, ycoord]:
        raise ValueError(f"Can not unroate coordinates for {cube}")

    # define reg latlon coordinate system and pair of locations
    # to be derived in the cube coordinate space.
    lat_lon_coord_system = iris.coord_systems.GeogCS(
        semi_major_axis=iris.fileformats.pp.EARTH_RADIUS
    )
    lons = np.array([MINLON, MAXLON], dtype=float)
    lats = np.array([MINLAT, MAXLAT], dtype=float)

    # Calculate pair location in cube coordinates
    if xcoord != "longitude" and ycoord != "latitude":

        if xcoord == "grid_longitude" and ycoord == "grid_latitude":
            # rotated coord ...
            pole_lon = cube.coord(xcoord).coord_system.grid_north_pole_longitude
            pole_lat = cube.coord(ycoord).coord_system.grid_north_pole_latitude
            
            # Perform rotation
            rot_lons, rot_lats = rotate_pole(lons, lats, pole_lon, pole_lat)
            rot_lons = rot_lons + 360
            
            print(f"\nReg Lat {lats} converted to " + \
                  f"rotated lat coord: {rot_lats}")
            print(f"Reg Lon {lons} converted to " + \
                  f"rotated lon coord: {rot_lons}")

            lat_constraint = iris.Constraint(
                grid_latitude=lambda cell: rot_lats[0] < cell < rot_lats[1]
                )
            lon_constraint = iris.Constraint(
                grid_longitude=lambda cell: rot_lons[0] < cell < rot_lons[1]
                )

            cube = cube.extract(lon_constraint & lat_constraint)

        elif (xcoord == "projection_x_coordinate" and 
              ycoord == "projection_y_coordinate"):
            # Other coordinate system (note this may work for x/ycoords other than
            # those considered here
            ll_crs = lat_lon_coord_system.as_cartopy_crs()
            cube_crs = cube.coord(xcoord).coord_system.as_cartopy_crs()

            # Convert to lat/lon points
            cube_lonlats = cube_crs.transform_points(ll_crs, lons, lats)
            cube_lons = cube_lonlats[:, 0]
            cube_lats = cube_lonlats[:, 1]

            print(f"\nReg Lat {lats} converted to " + \
                  f"projection_y_coordinate: {cube_lats}")
            print(f"Reg Lon {lons} converted to " + \
                  f"projection_x_coordinate: {cube_lons}")
            
            lat_constraint = iris.Constraint(
                projection_y_coordinate=lambda cell: cube_lats[0] < cell < cube_lats[1]
                )
            lon_constraint = iris.Constraint(
                projection_x_coordinate=lambda cell: cube_lons[0] < cell < cube_lons[1]
                )

            cube = cube.extract(lon_constraint & lat_constraint)

    return(cube)


###############################################################################
# precision_round:
###############################################################################

def precision_round(numbers, digits = 3):
    '''
    Parameters:
    -----------
    numbers : scalar, 1D , or 2D array(-like)
    digits: number of digits after decimal point
    
    Returns:
    --------
    out : same shape as numbers
    '''
    import numpy as np

    numbers = np.asarray(np.atleast_2d(numbers))
    out_array = np.zeros(numbers.shape) # the returning array
    
    for dim0 in range(numbers.shape[0]):
        powers = [int(F"{number:e}".split('e')[1]) for number in numbers[dim0, :]]
        out_array[dim0, :] = [round(number, -(int(power) - digits))
                         for number, power in zip(numbers[dim0, :], powers)]
        
    # returning the original shape of the `numbers` 
    if out_array.shape[0] == 1 and out_array.shape[1] == 1:
        out_array = out_array[0, 0]
    elif out_array.shape[0] == 1:
        out_array = out_array[0, :]
    
    return out_array


###############################################################################
# add_bounds: 
#
# Adding bounds to projection 
# with grid_latitude/grid_longitude coordinates.
###############################################################################

def add_bounds(cube):
    
    import iris
    
    if not cube.coord('grid_latitude').has_bounds():
        cube.coord('grid_latitude').guess_bounds()
        #print("adding grid_latitude bound")
    if not cube.coord('grid_longitude').has_bounds():
        cube.coord('grid_longitude').guess_bounds()
        #print("adding grid_longitude bound")
    
    return cube


###############################################################################
# calc_forecast_day:
#
# Copied from adaq python librairy
###############################################################################

def calc_forecast_day(forecast_period, runtime, day_start_hour=None):

    import numpy as np

    # Set up hour of day to refer to as the start of a day
    # Note for hourly means, the convention is for the coordinate point to be
    # placed at the end of the meaning period, hence by default day_start_hour
    # is set to 1, which is equivalent to 0Z-1Z mean period.
    # However if runtime is 12Z, then for historical reasons, the first 'Day'
    # is counted as 12-13Z -> 11-12Z.
    # Note these defaults can be overridden with the keyword day_start_hour.
    if day_start_hour is None:
        day_start_hour = 1
        if runtime == 12:
            day_start_hour = 13

    # Number of hours from runtime to first day_start_hour:
    nhrs_to_start_of_day = (day_start_hour - runtime) % 24
    # Number of hours after first day_start_hour:
    nhrs_after_start_of_first_day = np.array(forecast_period) - np.array(
        nhrs_to_start_of_day
    )
    # Number of days after first day_start_hour:
    ndays_after_start_of_first_day = nhrs_after_start_of_first_day // 24
    # But day_start_hour should be referred to as 'Day1', so add an extra day on
    forecast_day = ndays_after_start_of_first_day + 1

    # Convert back to same type as input:
    if isinstance(forecast_period, int):
        forecast_day = int(forecast_day)
    elif isinstance(forecast_period, list):
        forecast_day = list(forecast_day)
    elif isinstance(forecast_day, np.ndarray):
        # Ensure all integers
        forecast_day = forecast_day.astype(int)

    return forecast_day


###############################################################################
#         coord_list = cube.coords()
###############################################################################

def __callback_add_forecast_day(cube, field, filename):

    import iris

    #print(cube.name())

    coord_list = cube.coords()
    
    for coord in coord_list:
        if coord.standard_name == "forecast_reference_time":
            runtime_hr = coord.units.num2date(coord.points[0]).hour
            cube.remove_coord("forecast_reference_time")

    if not cube.coords("forecast_day"):
        forecast_period = cube.coord("forecast_period").points[0]
        forecast_day = calc_forecast_day(
            forecast_period, runtime_hr
        )
        cube.add_aux_coord(
            iris.coords.DimCoord(
                forecast_day, long_name="forecast_day", units="Days"
            )
        )


    return cube
