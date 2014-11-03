def kwh_to_BTU(kwh):
    """
    Return kwh value in BTUs
    """
    return 3.41214163e3 * kwh

def BTU_to_kwh(BTU):
    """
    Return BTU value in kwhs
    """
    return 2.9307107e-4 * BTU

def therm_to_BTU(therm):
    """
    Return therm value in BTUs
    """
    return 9.9976129e4 * therm

def BTU_to_therm(BTU):
    """
    Return BTU value in therms
    """
    return 1.00023877e-5 * BTU

def kwh_to_therm(kwh):
    """
    Return kwh value in therms
    """
    return kwh * 0.0341295634

def therm_to_kwh(therm):
    """
    Return therm value in kwhs
    """
    return therm * 29.3001111

def farenheight_to_celsius(f):
    """
    Return Farenheight value in Celsius
    """
    return (5./9) * (f-32)

def celsius_to_farenheight(c):
    """
    Return Celsius value in Farenheight
    """
    return (9./5)*c + 32

def temp_to_hdd(t,base):
    """
    Return temperature value (any units) in heating degree days (HDD_base)
    """
    if t >= base:
        return 0
    return base - t

def temp_to_cdd(t,base):
    """
    Return temperature value (any units) in cooling degree days (CDD_base)
    """
    if t <= base:
        return 0
    return t - base
