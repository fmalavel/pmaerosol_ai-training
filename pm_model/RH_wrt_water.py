def RH_wrt_water_from_file(infile, theta=False, sect30_T=False):

    # Uses the "Lowe & Ficke" equation (at the back of the second edition of
    # Pruppacher & Klett's magnum opus) to calculate saturation vapour pressure
    # with respect to water (T in degC is the only input).
    # It then uses equations 2.18 and 2.19 of Rogers & Yau (3rd Ed) to convert
    # the UM's specific humidity (q) into mixing ratio (w), and 2.18 again with
    # e formally set to esat to calculate wsat.
    # Finally, following equation 2.20, RH is calulated as 100*w/wsat.
    #
    # Note that values for cold temperatuers (< -50degC) are nonsense, as are
    # values above the tropopause: use RH_wrt_ice.py instead.
    #
    # NOTE: As of 24/5/21 field 16-004 is assumed for temperature rather than
    #       30-111 as used previously. If the use of the latter is required,
    #       set "sect30_T=True" in the call.
 
    import iris
    import cf_units
    import numpy.ma as ma
    import theta_to_T
 
    # Load specific humidity (kg/kg-moist-air), pressure (Pa) and either
    # temperature or potential temperature (K):
    q = iris.load_cube(infile, iris.AttributeConstraint(STASH='m01s00i010'))
    p = iris.load_cube(infile, iris.AttributeConstraint(STASH='m01s00i408'))
    if theta:
       T = theta_to_T.theta_to_T(infile)
    else:
       if sect30_T:
          T = iris.load_cube(infile,
                                iris.AttributeConstraint(STASH='m01s30i111'))
       else:
          T = iris.load_cube(infile,
                                iris.AttributeConstraint(STASH='m01s16i004'))
 
    T_C = T - 273.15
 
    a0 = 6.107799961
    a1 = 4.436518521e-01
    a2 = 1.428945805e-02
    a3 = 2.650648471e-04
    a4 = 3.031240396e-06
    a5 = 2.034080948e-08
    a6 = 6.136820929e-11
 
    esat_mb = a0+(T_C*(a1+(T_C*(a2+(T_C*(a3+(T_C*(a4+(T_C*(a5+(T_C*a6)))))))))))
    esat_Pa = esat_mb * 100.0
    esat_Pa.units = cf_units.Unit('Pa')
 
    eps = 0.622
 
    w = q * (p-((1.0-eps)*esat_Pa)) / (p-esat_Pa)
 
    wsat = eps * esat_Pa/(p-esat_Pa)
 
    RHw = 100.0 * w/wsat
 
    data_masked = ma.masked_where(T_C.data < -50.0, RHw.data)
    RHw.data = data_masked
 
    return RHw


def RH_wrt_water_from_cubes(theta=None,p=None,q=None):

    # Uses the "Lowe & Ficke" equation (at the back of the second edition of
    # Pruppacher & Klett's magnum opus) to calculate saturation vapour pressure
    # with respect to water (T in degC is the only input).
    # It then uses equations 2.18 and 2.19 of Rogers & Yau (3rd Ed) to convert
    # the UM's specific humidity (q) into mixing ratio (w), and 2.18 again with
    # e formally set to esat to calculate wsat.
    # Finally, following equation 2.20, RH is calulated as 100*w/wsat.
    #
    # Note that values for cold temperatuers (< -50degC) are nonsense, as are
    # values above the tropopause: use RH_wrt_ice.py instead.
 
    import iris
    import cf_units
    import numpy.ma as ma
    
    # TODO add checks to verify that STASH codes of input cubes matches this:
        
    # q = cubelist.extract_cube(iris.AttributeConstraint(STASH='m01s00i010'))
    # p = cubelist.extract_cube(iris.AttributeConstraint(STASH='m01s00i408'))
    # theta = cubelist.extract_cube(iris.AttributeConstraint(STASH='m01s00i004'))
 
    q = q
    p = p
    theta = theta
    
    # Theta -> ambiant T
    R = 287.050
    cP = 1005.00
    pref = 100000.0
    p.units = cf_units.Unit(None)     # Iris won't raise units to a non-integer 
                                      # power, for ****'s sake....
    T = theta * (p/pref)**(R/cP)
    T.units = cf_units.Unit('K')
     
    T_zeroK = 273.15
    T_C = T - T_zeroK
 
    a0 = 6.107799961
    a1 = 4.436518521e-01
    a2 = 1.428945805e-02
    a3 = 2.650648471e-04
    a4 = 3.031240396e-06
    a5 = 2.034080948e-08
    a6 = 6.136820929e-11
 
    esat_mb = a0+(T_C*(a1+(T_C*(a2+(T_C*(a3+(T_C*(a4+(T_C*(a5+(T_C*a6)))))))))))
    esat_Pa = esat_mb * 100.0
    esat_Pa.units = cf_units.Unit('Pa')
 
    eps = 0.622
 
    p.units = cf_units.Unit('Pa')    # reintroduce p units to allow substraction
    w = q * (p-((1.0-eps)*esat_Pa)) / (p-esat_Pa)
 
    wsat = eps * esat_Pa/(p-esat_Pa)
 
    RHw = 100.0 * w/wsat
 
    data_masked = ma.masked_where(T_C.data < -50.0, RHw.data)
    RHw.data = data_masked
    RHw.units = '%'
 
    return RHw
