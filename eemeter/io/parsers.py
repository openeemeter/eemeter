from collections import defaultdict
from datetime import datetime, timedelta
from lxml import etree
import pytz
import six
import warnings

from eemeter.structures import EnergyTrace
from eemeter.io.serializers import ArbitrarySerializer


class ESPIUsageParser(object):
    """ Parse ESPI XML files.

    Basic usage:

    .. code-block:: python

        >>> from eemeter.io.parsers import ESPIUsageParser
        >>> with open("/path/to/example.xml") as f:
        ...     parser = ESPIUsageParser(f)
        >>> energy_traces = list(parser.get_energy_traces())

    Parameters
    ----------
    xml : str, filepath, file buffer
        XML data to parse
    """

    SERVICE_KIND = {
        '0': 'electricity',
        '1': 'gas',
        '2': 'water',
        '3': 'time',
        '4': 'heat',
        '5': 'refuse',
        '6': 'sewerage',
        '7': 'rates',
        '8': 'tvLicense',
        '9': 'internet',
    }

    QUALITY_OF_READING = {
        '0': 'valid',
        '7': 'manually edited',
        '8': 'estimated using reference day',
        '9': 'estimated using linear interpolation',
        '10': 'questionable',
        '11': 'derived',
        '12': 'projected (forecast)',
        '13': 'mixed',
        '14': 'raw',
        '15': 'normalized for weather',
        '16': 'other',
        '17': 'validated',
        '18': 'verified',
        '19': 'revenue-quality',
    }

    ACCUMULATION_KIND = {
        '0': 'none',
        '1': 'bulkQuantity',
        '2': 'continuousCumulative',
        '3': 'cumulative',
        '4': 'deltaData',
        '6': 'indicating',
        '9': 'summation',
        '10': 'timeDelta',
        '12': 'instantaneous',
        '13': 'latchingQuantity',
        '14': 'boundedQuantity',
    }

    COMMODITY_KIND = {
        '0': 'none',
        '1': 'electricity SecondaryMetered',
        '2': 'electricity PrimaryMetered',
        '3': 'communication',
        '4': 'air',
        '5': 'insulativeGas',
        '6': 'insulativeOil',
        '7': 'naturalGas',
        '8': 'propane',
        '9': 'potableWater',
        '10': 'steam',
        '11': 'wasteWater',
        '12': 'heatingFluid',
        '13': 'coolingFluid',
        '14': 'nonpotableWater',
        '15': 'nox',
        '16': 'so2',
        '17': 'ch4',
        '18': 'co2',
        '19': 'carbon',
        '20': 'ch4',
        '21': 'pfc',
        '22': 'sf6',
        '23': 'tvLicense',
        '24': 'internet',
        '25': 'refuse',
    }

    DATA_QUALIFIER_KIND = {
        '0': 'none',
        '2': 'average',
        '4': 'excess',
        '5': 'highThreshold',
        '7': 'lowThreshold',
        '8': 'maximum',
        '9': 'minimum',
        '11': 'nominal',
        '12': 'normal',
        '16': 'secondMaximum',
        '17': 'secondMinimum',
        '23': 'thirdMaximum',
        '24': 'fourthMaximum',
        '25': 'fifthMaximum',
        '26': 'sum',
    }

    FLOW_DIRECTION_KIND = {
        '0': 'none',
        '1': 'forward',
        '2': 'lagging',
        '3': 'leading',
        '4': 'net',
        '5': 'q1plusQ2',
        '7': 'q1plusQ3',
        '8': 'q1plusQ4',
        '9': 'q1minusQ4',
        '10': 'q2plusQ3',
        '11': 'q2plusQ4',
        '12': 'q2minusQ3',
        '13': 'q3plusQ4',
        '14': 'q3minusQ2',
        '15': 'quadrant1',
        '16': 'quadrant2',
        '17': 'quadrant3',
        '18': 'quadrant4',
        '19': 'reverse',
        '20': 'total',
        '21': 'totalByPhase',
    }

    MEASUREMENT_KIND = {
        '0': 'none',
        '2': 'apparentPowerFactor',
        '3': 'currency',
        '4': 'current',
        '5': 'currentAngle',
        '6': 'currentImbalance',
        '7': 'date',
        '8': 'demand',
        '9': 'distance',
        '10': 'distortionVoltAmperes',
        '11': 'energization',
        '12': 'energy',
        '13': 'energizationLoadSide',
        '14': 'fan',
        '15': 'frequency',
        '16': 'funds',
        '17': 'ieee1366ASAI',
        '18': 'ieee1366ASIDI',
        '19': 'ieee1366ASIFI',
        '20': 'ieee1366CAIDI',
        '21': 'ieee1366CAIFI',
        '22': 'ieee1366CEMIn',
        '23': 'ieee1366CEMSMIn',
        '24': 'ieee1366CTAIDI',
        '25': 'ieee1366MAIFI',
        '26': 'ieee1366MAIFIe',
        '27': 'ieee1366SAIDI',
        '28': 'ieee1366SAIFI',
        '31': 'lineLosses',
        '32': 'losses',
        '33': 'negativeSequence',
        '34': 'phasorPowerFactor',
        '35': 'phasorReactivePower',
        '36': 'positiveSequence',
        '37': 'power',
        '38': 'powerFactor',
        '40': 'quantityPower',
        '41': 'sag',
        '42': 'swell',
        '43': 'switchPosition',
        '44': 'tapPosition',
        '45': 'tariffRate',
        '46': 'temperature',
        '47': 'totalHarmonicDistortion',
        '48': 'transformerLosses',
        '49': 'unipedeVoltageDip10to15',
        '50': 'unipedeVoltageDip15to30',
        '51': 'unipedeVoltageDip30to60',
        '52': 'unipedeVoltageDip60to90',
        '53': 'unipedeVoltageDip90to100',
        '54': 'voltage',
        '55': 'voltageAngle',
        '56': 'voltageExcursion',
        '57': 'voltageImbalance',
        '58': 'volume',
        '59': 'zeroFlowDuration',
        '60': 'zeroSequence',
        '64': 'distortionPowerFactor',
        '81': 'frequencyExcursion',
        '90': 'applicationContext',
        '91': 'apTitle',
        '92': 'assetNumber',
        '93': 'bandwidth',
        '94': 'batteryVoltage',
        '95': 'broadcastAddress',
        '96': 'deviceAddressType1',
        '97': 'deviceAddressType2',
        '98': 'deviceAddressType3',
        '99': 'deviceAddressType4',
        '100': 'deviceClass',
        '101': 'electronicSerialNumber',
        '102': 'endDeviceID',
        '103': 'groupAddressType1',
        '104': 'groupAddressType2',
        '105': 'groupAddressType3',
        '106': 'groupAddressType4',
        '107': 'ipAddress',
        '108': 'macAddress',
        '109': 'mfgAssignedConfigurationID',
        '110': 'mfgAssignedPhysicalSerialNumber',
        '111': 'mfgAssignedProductNumber',
        '112': 'mfgAssignedUniqueCommunicationAddress',
        '113': 'multiCastAddress',
        '114': 'oneWayAddress',
        '115': 'signalStrength',
        '116': 'twoWayAddress',
        '117': 'signaltoNoiseRatio',
        '118': 'alarm',
        '119': 'batteryCarryover',
        '120': 'dataOverflowAlarm',
        '121': 'demandLimit',
        '122': 'demandReset',
        '123': 'diagnostic',
        '124': 'emergencyLimit',
        '125': 'encoderTamper',
        '126': 'ieee1366MomentaryInterruption',
        '127': 'ieee1366MomentaryInterruptionEvent',
        '128': 'ieee1366SustainedInterruption',
        '129': 'interruptionBehaviour',
        '130': 'inversionTamper',
        '131': 'loadInterrupt',
        '132': 'loadShed',
        '133': 'maintenance',
        '134': 'physicalTamper',
        '135': 'powerLossTamper',
        '136': 'powerOutage',
        '137': 'powerQuality',
        '138': 'powerRestoration',
        '139': 'programmed',
        '140': 'pushbutton',
        '141': 'relayActivation',
        '142': 'relayCycle',
        '143': 'removalTamper',
        '144': 'reprogrammingTamper',
        '145': 'reverseRotationTamper',
        '146': 'switchArmed',
        '147': 'switchDisabled',
        '148': 'tamper',
        '149': 'watchdogTimeout',
        '150': 'billLastPeriod',
        '151': 'billToDate',
        '152': 'billCarryover',
        '153': 'connectionFee',
        '154': 'audibleVolume',
        '155': 'volumetricFlow',
    }

    TIME_ATTRIBUTE_KIND = {
        "0": "none",
        "1": "tenMinute",
        "2": "fifteenMinute",
        "3": "oneMinute",
        "4": "twentyfourHour",
        "5": "thirtyMinute",
        "6": "fiveMinute",
        "7": "sixtyMinute",
        "10": "twoMinute",
        "14": "threeMinute",
        "15": "present",
        "16": "previous",
        "31": "twentyMinute",
        "50": "fixedBlock60Min",
        "51": "fixedBlock30Min",
        "52": "fixedBlock20Min",
        "53": "fixedBlock15Min",
        "54": "fixedBlock10Min",
        "55": "fixedBlock5Min",
        "56": "fixedBlock1Min",
        "57": "rollingBlock60MinIntvl30MinSubIntvl",
        "58": "rollingBlock60MinIntvl20MinSubIntvl",
        "59": "rollingBlock60MinIntvl15MinSubIntvl",
        "60": "rollingBlock60MinIntvl12MinSubIntvl",
        "61": "rollingBlock60MinIntvl10MinSubIntvl",
        "62": "rollingBlock60MinIntvl6MinSubIntvl",
        "63": "rollingBlock60MinIntvl5MinSubIntvl",
        "64": "rollingBlock60MinIntvl4MinSubIntvl",
        "65": "rollingBlock30MinIntvl15MinSubIntvl",
        "66": "rollingBlock30MinIntvl10MinSubIntvl",
        "67": "rollingBlock30MinIntvl6MinSubIntvl",
        "68": "rollingBlock30MinIntvl5MinSubIntvl",
        "69": "rollingBlock30MinIntvl3MinSubIntvl",
        "70": "rollingBlock30MinIntvl2MinSubIntvl",
        "71": "rollingBlock15MinIntvl5MinSubIntvl",
        "72": "rollingBlock15MinIntvl3MinSubIntvl",
        "73": "rollingBlock15MinIntvl1MinSubIntvl",
        "74": "rollingBlock10MinIntvl5MinSubIntvl",
        "75": "rollingBlock10MinIntvl2MinSubIntvl",
        "76": "rollingBlock10MinIntvl1MinSubIntvl",
        "77": "rollingBlock5MinIntvl1MinSubIntvl",
    }

    UNIT_SYMBOL_KIND = {
        "0": "none",
        "2": "m",
        "3": "g",
        "4": "revPerSec",
        "5": "A",
        "6": "K",
        "7": "mol",
        "8": "cd",
        "9": "deg",
        "10": "rad",
        "11": "sr",
        "21": "gy",
        "22": "bq",
        "23": "degC",
        "24": "sv",
        "25": "F",
        "27": "sec",
        "28": "H",
        "29": "V",
        "30": "ohm",
        "31": "J",
        "32": "n",
        "33": "Hz",
        "34": "lx",
        "35": "lm",
        "36": "wb",
        "37": "t",
        "38": "W",
        "39": "pa",
        "41": "m2",
        "42": "m3",
        "43": "mPerSec",
        "44": "mPerSec2",
        "45": "m3PerSec",
        "46": "mPerM3",
        "47": "kgM",
        "48": "kgPerM3",
        "49": "m2PerSec",
        "50": "wPerMK",
        "51": "jPerK",
        "53": "siemens",
        "54": "radPerSec",
        "61": "VA",
        "63": "VAr",
        "65": "cosTheta",
        "66": "Vs",
        "67": "V2",
        "68": "As",
        "69": "A2",
        "70": "A2s",
        "71": "VAh",
        "72": "Wh",
        "73": "VArh",
        "74": "VPerHz",
        "75": "HzPerSec",
        "76": "char",
        "77": "charPerSec",
        "78": "gM2",
        "79": "b",
        "80": "money",
        "81": "WPerSec",
        "82": "litrePerSec",
        "100": "q",
        "101": "qh",
        "102": "ohmM",
        "103": "APerM",
        "104": "V2h",
        "105": "A2h",
        "106": "Ah",
        "107": "WhPerM3",
        "108": "timeStamp",
        "109": "status",
        "111": "count",
        "113": "bm",
        "114": "code",
        "115": "WhPerRev",
        "116": "VArhPerRev",
        "117": "VAhPerRev",
        "118": "meCode",
        "119": "ft3",
        "120": "ft3compensated",
        "123": "ft3compensatedPerH",
        "125": "m3PerH",
        "126": "m3compensatedPerH",
        "127": "m3uncompensatedPerH",
        "128": "usGal",
        "129": "usGalPerH",
        "130": "imperialGal",
        "131": "imperialGalPerH",
        "132": "btu",
        "133": "btuPerH",
        "134": "litre",
        "137": "litrePerH",
        "138": "litreCompensatedPerH",
        "139": "litreUncompensatedPerH",
        "140": "paG",
        "141": "psiA",
        "142": "psiG",
        "143": "litrePerLitre",
        "144": "gPerG",
        "145": "molPerM3",
        "146": "molPerMol",
        "147": "molPerKg",
        "148": "mPerM",
        "149": "secPerSec",
        "150": "HzPerHz",
        "151": "VPerV",
        "152": "APerA",
        "153": "WPerVA",
        "154": "rev",
        "155": "paA",
        "156": "litreUncompensated",
        "157": "litreCompensated",
        "158": "kat",
        "159": "min",
        "160": "h",
        "161": "q45",
        "162": "q60",
        "163": "q45h",
        "164": "q60h",
        "165": "jPerKg",
        "166": "m3uncompensated",
        "167": "m3compensated",
        "168": "WPerW",
        "169": "therm",
    }

    VALUE_PARSERS = {
        '{http://naesb.org/espi}accumulationBehaviour': ACCUMULATION_KIND.get,
        '{http://naesb.org/espi}commodity': COMMODITY_KIND.get,
        '{http://naesb.org/espi}dataQualifier': DATA_QUALIFIER_KIND.get,
        '{http://naesb.org/espi}defaultQuality': QUALITY_OF_READING.get,
        '{http://naesb.org/espi}flowDirection': FLOW_DIRECTION_KIND.get,
        '{http://naesb.org/espi}intervalLength':
            lambda x: timedelta(seconds=int(x)),
        '{http://naesb.org/espi}kind': MEASUREMENT_KIND.get,
        '{http://naesb.org/espi}powerOfTenMultiplier': lambda x: int(x),
        '{http://naesb.org/espi}timeAttribute': TIME_ATTRIBUTE_KIND.get,
        '{http://naesb.org/espi}uom': UNIT_SYMBOL_KIND.get,
        '{http://naesb.org/espi}measuringPeriod': TIME_ATTRIBUTE_KIND.get
    }

    def __init__(self, xml):
        try:
            self.root = etree.parse(xml)  # xml is file path or file object
        except IOError:
            if isinstance(xml, six.string_types + (six.binary_type,)):
                self.root = etree.fromstring(xml)  # xml is a string.
        self.timezone = self._get_timezone()

    def has_solar(self):
        """ Returns True if there is a "reverse" flow direction in this file,
        indicating presence of solar photo voltaics.

        TODO: Verify that this is the correct way to determine this - are there
        false positives or false negatives? Is there a more straightforward
        flag to use somewhere else?
        """
        reading_type_elements = \
            self.root.findall('.//{http://naesb.org/espi}ReadingType')
        reading_types = [
            self._parse_reading_type(e)
            for e in reading_type_elements
        ]
        flow_directions = [rt["flow_direction"] for rt in reading_types]
        return "reverse" in flow_directions

    def _normalize_fuel_type(self, commodity):
        ''' Convert ESPI fuel type codes to eemeter fuel type codes.
        '''
        fuel_types = {
            "naturalGas": "natural_gas",
            "electricity SecondaryMetered": "electricity"
        }
        try:
            return fuel_types[commodity]
        except KeyError:
            return commodity

    def _tz_offset_to_timezone(self, tz_offset):
        ''' Convert ESPI timezone offset code to python timezone object.'''
        if tz_offset == "-28800":
            return pytz.timezone("US/Pacific")
        elif tz_offset == "-25200":
            return pytz.timezone("US/Mountain")
        elif tz_offset == "-21600":
            return pytz.timezone("US/Central")
        elif tz_offset == "-18000":
            return pytz.timezone("US/Eastern")
        else:
            raise ValueError("Timezone not supported")

    def _get_timezone(self):
        ''' Fetch the timezone the interval readings are in, from
        the ESPI LocalTimeParameters object.

        Returns
        -------
        timezone : datetime tzinfo
            Timezone info as recognized by python datetime objects.
        '''
        local_time_parameters = self.root.find(
            './/{http://naesb.org/espi}LocalTimeParameters')

        try:
            # Parse Daylight Savings Time elements.
            #   The start rule and end rule are weird encoded ways of saying
            #   when DST should be in effect, and the offset is the actual
            #   effect.
            dst_start_rule = local_time_parameters.find(
                '{http://naesb.org/espi}dstStartRule').text
            dst_end_rule = local_time_parameters.find(
                '{http://naesb.org/espi}dstEndRule').text
            dst_offset = local_time_parameters.find(
                '{http://naesb.org/espi}dstOffset').text
        except AttributeError:
            # one or more of these was not found - assume UTC.
            return pytz.UTC
        else:
            # Check that timezone is a standard timezone.
            #   Non-standard timezones might not have DST attributes,
            #   break loudly if you encounter them.
            assert dst_start_rule == "360E2000"
            assert dst_end_rule == "B40E2000"
            assert dst_offset == "3600"

        # Find the ESPI timezone offset code, and convert it to
        # a python timezone object.
        tz_offset = local_time_parameters.find(
            '{http://naesb.org/espi}tzOffset').text
        return self._tz_offset_to_timezone(tz_offset)

    class _ChildElementGetter(object):
        ''' Helper class that gets child (or really descendant) elements
        of given element, extract their text values, and parses them.

        Parameters
        ----------
        element : etree.element
            Element within which to find child elements
        value_parsers : dict of callable
            Callables keyed by element name that take element text and return
            an object representing the element's value.
        '''
        def __init__(self, element, value_parsers):
            self.element = element
            # Different child elements have different value parsing functions.
            self.VALUE_PARSERS = value_parsers

        def child_element_value(self, child_element_name):
            '''Return parsed text value of child element.

            Parameters
            ----------
            child_element_name : str
                name of child element for which to find the value.
                E.g. '{http://naesb.org/espi}kind'

            Returns
            -------
            child_element_value : object
                Value of child element as parsed by the known set of value
                parsers.
            '''
            child_element = self.element.find(child_element_name)
            if child_element is not None:
                try:
                    return self.VALUE_PARSERS[child_element_name](
                        child_element.text)
                except KeyError:
                    msg = (
                        'No parsing function defined for text value of'
                        ' element %s' % child_element_name
                    )
                    raise NotImplementedError(msg)

    def _parse_reading_type(self, reading_type_element):
        """ Parses a single reading type element.

        Parameters
        ----------
        reading_type_element : etree.Element
            ReadingType element for which to get data.

        Returns
        -------
        data : dict
            Data in the ReadingType element.
        """

        # Initialize Getter class for reading type element, to make getting
        # and parsing the values of child elements easier.
        reading_type = self._ChildElementGetter(
            reading_type_element, self.VALUE_PARSERS)

        data_spec = [
            ('accumulation_behavior',
                '{http://naesb.org/espi}accumulationBehaviour'),
            ('commodity', '{http://naesb.org/espi}commodity'),
            ('data_qualifier', '{http://naesb.org/espi}dataQualifier'),
            ('default_quality', '{http://naesb.org/espi}defaultQuality'),
            ('flow_direction', '{http://naesb.org/espi}flowDirection'),
            ('interval_length', '{http://naesb.org/espi}intervalLength'),
            ('kind', '{http://naesb.org/espi}kind'),
            ('power_of_ten_multiplier',
                '{http://naesb.org/espi}powerOfTenMultiplier'),
            ('time_attribute', '{http://naesb.org/espi}timeAttribute'),
            ('uom', '{http://naesb.org/espi}uom'),
            ('measuring_period', '{http://naesb.org/espi}measuringPeriod'),
        ]

        return {
            name: reading_type.child_element_value(path)
            for name, path in data_spec
        }

    def _get_reading_type_interval_block_groups(self):
        """ Yields reading type elements and their associated interval blocks.

        Parameters
        ----------
        reading_type_element : etree.Element
            ReadingType element for which to get data.

        Yields
        -------
        data : dict
            JSON-like representation of interval blocks.

        """

        # Want to be able to pick off a MeterReading element out of sequence
        # when we hit a ReadingType, so use an iterator,
        entry_elements = self.root.findall(
            './{http://www.w3.org/2005/Atom}entry')

        def _reading_type_element(entry):
            return entry.find(".//{http://naesb.org/espi}ReadingType")

        def _interval_block_element(entry):
            return entry.find(".//{http://naesb.org/espi}IntervalBlock")

        def _meter_reading_element(entry):
            return entry.find(".//{http://naesb.org/espi}MeterReading")

        reading_types = {}

        recent_reading_type = None

        for entry in entry_elements:

            interval_block_element = _interval_block_element(entry)

            if interval_block_element is None:
                reading_type_element = _reading_type_element(entry)
                meter_reading_element = _meter_reading_element(entry)
                if reading_type_element is not None:
                    recent_reading_type = reading_type_element
                elif meter_reading_element is not None:
                    if recent_reading_type is not None:
                        # why doesn't reading type have this id?
                        meter_reading_id = (
                            entry.getchildren()[2]
                            .attrib["href"].split('/')[-1]
                        )
                        reading_types[meter_reading_id] = {
                            "reading_type": recent_reading_type,
                            "interval_blocks": [],
                        }
                        recent_reading_type = None
                else:
                    # ignore other types, like UsagePoint, which contain
                    # reduntant info
                    pass
            else:
                try:
                    meter_reading_id = (
                        entry.getchildren()[1]
                        .attrib["href"].split('/')[-2]
                    )
                except:
                    pass
                else:
                    reading_types[meter_reading_id]["interval_blocks"] \
                        .append(interval_block_element)
                    continue

                if meter_reading_id not in reading_types:
                    message = (
                        "Could not find the ReadingType for the IntervalBlock"
                        " element {} using the MeterReading ID {}."
                        .format(etree.tostring(entry), meter_reading_id)
                    )
                    warnings.warn(message)

        for group in reading_types.values():
            yield self._parse_interval_block_group(group)

    def _parse_interval_reading(self, interval_reading):
        '''
        Parse ESPI IntervalReading element into dict.

        IntervalReadings contain the core data observations that
        drive the eemeter: interval energy use measurements.

        This method uses document-level timezone attribute to
        correctly parse interval start times into tz-aware datetimes.

        Parameters
        ----------
        interval_reading : etree.Element
            IntervalReading element for which to get data.

        Returns
        -------
        data : dict
            Data in the IntervalReading element.
        '''
        reading_quality_element = interval_reading.find(
            "{http://naesb.org/espi}ReadingQuality/"
            "{http://naesb.org/espi}quality")
        try:
            reading_quality = \
                self.QUALITY_OF_READING[reading_quality_element.text]
        except AttributeError:
            reading_quality = None

        duration_element = interval_reading.find(
            "{http://naesb.org/espi}timePeriod/"
            "{http://naesb.org/espi}duration")
        duration = timedelta(seconds=int(duration_element.text))

        start_element = interval_reading.find(
            "{http://naesb.org/espi}timePeriod/"
            "{http://naesb.org/espi}start")

        # Timestamps are, by definition, UTC.
        # We ignore the given self.timezone, because we would convert to
        # UTC anyway
        start = datetime.fromtimestamp(int(start_element.text), tz=pytz.UTC)

        value = int(interval_reading.find("{http://naesb.org/espi}value").text)

        return {
            "reading_quality": reading_quality,
            "duration": duration,
            "start": start,
            "value": value
        }

    def _parse_interval_block_group(self, interval_block_group):
        '''
        Parse IntervalBlocks (and child IntervalReadings) grouped by
        ReadingType into dict.

        IntervalBlocks typically hold 24-hours worth of IntervalReadings,
        except at start and end points of the series, and at timezone
        transitions such as Daylight Savings time, during which 24-hour periods
        will be broken up. Some providers group all IntervalReadings into a
        single IntervalBlock.

        In addition interval readings, return the start and duration of the
        block, and a sibling ReadingType element which describes the block's
        readings.

        Parameters
        ----------
        interval_block_group : dict
            IntervalBlock elements, and associated ReadingType, e.g.::

                {
                    'reading_type': <Element ReadingType>},
                    'interval_blocks': [
                        <Element IntervalBlock>,
                        <Element IntervalBlock>,
                        <Element IntervalBlock>,
                        ...
                    ]
                }

        Returns
        -------
        data : dict
            Data in the group of IntervalBlock elements
        '''
        reading_type = self._parse_reading_type(
            interval_block_group["reading_type"])

        interval_blocks = interval_block_group["interval_blocks"]

        return {
            "reading_type": reading_type,
            "interval_blocks": [
                self._parse_interval_block(interval_block)
                for interval_block in interval_blocks
            ],
        }

    def _parse_interval_block(self, interval_block):
        '''
        Parameters
        ----------
        interval_block : dict
            IntervalBlock element to parse

        Returns
        -------
        data : dict
            Data in the IntervalBlock element.
        '''
        # Capture start and duration of the interval block.
        interval_duration_element = interval_block.find(
            "{http://naesb.org/espi}interval/"
            "{http://naesb.org/espi}duration")
        interval_start_element = interval_block.find(
            "{http://naesb.org/espi}interval/"
            "{http://naesb.org/espi}start")
        interval_duration = timedelta(
            seconds=int(interval_duration_element.text))
        interval_start = datetime.fromtimestamp(
            int(interval_start_element.text), tz=pytz.UTC)

        # Collect and parse all interval readings for the block.
        interval_reading_elements = interval_block.findall(
            "{http://naesb.org/espi}IntervalReading")
        interval_readings = [
            self._parse_interval_reading(reading)
            for reading in interval_reading_elements
        ]

        data = {
            "interval": {
                "duration": interval_duration,
                "start": interval_start
            },
            "interval_readings": interval_readings
        }

        return data

    def _get_interval_block_group_consumption_records(self,
                                                      interval_block_group):
        ''' Return all  in ESPI Energy Usage XML.
        Each interval block contains a set of interval readings.

        Yields
        ------
        record : dict
            IntervalBlock record with start and end dates, e.g.::

                data = {
                    "start": datetime(2012, 1, 1, 0, 0, 0, tzinfo=pytz.UTC),
                    "end": datetime(2012, 1, 1, 0, 15, 0, tzinfo=pytz.UTC),
                    "value": 0.0,
                    "estimated": True,
                    "fuel_type": "electricity",
                    "unit_name": "kWh",
                }
        '''

        # Values must be adjusted with interval-block level multiplier.
        # Package block readings with adjusted units and block fuel type.
        fuel_type = self._normalize_fuel_type(
            interval_block_group["reading_type"]["commodity"])
        multiplier = 10 ** interval_block_group["reading_type"][
            "power_of_ten_multiplier"]
        unit_name = interval_block_group["reading_type"]["uom"]

        for interval_block in interval_block_group["interval_blocks"]:

            # For validation - see below
            total_duration = interval_block["interval"]["duration"]
            total_duration_s = total_duration.total_seconds()
            summed_durations = 0

            for interval_reading in interval_block["interval_readings"]:

                reading_quality = interval_reading["reading_quality"]
                if reading_quality is None:
                    estimated = False  # assume not estimated
                else:
                    estimated = "estimated" in reading_quality

                duration = interval_reading["duration"]
                summed_durations += duration.total_seconds()
                data = {
                    "start": interval_reading["start"],
                    "end": interval_reading["start"] + duration,
                    "value": interval_reading["value"] * multiplier,
                    "estimated": estimated,
                    "fuel_type": fuel_type,
                    "unit_name": unit_name
                }
                yield data

            # Validates that total interval block duration matches sum of
            # interval reading durations
            if not total_duration_s == summed_durations:
                message = (
                    "Total IntervalBlock duration != "
                    "  sum of component IntervalReading durations\n"
                    "  {}s != {}s"
                    .format(total_duration_s, summed_durations)
                )
                warnings.warn(message)

    def _get_consumption_record_groups(self):
        ''' Return all consumption records, across all IntervalBlocks,
        stored in ESPI Energy Usage XML, grouped by flow direction.

        Yields
        ------
        interval_reading : dict
            IntervalReading data
        '''

        for group in self._get_reading_type_interval_block_groups():
            flow_direction = group["reading_type"]["flow_direction"]
            records = list(
                self._get_interval_block_group_consumption_records(group))
            yield flow_direction, sorted(records, key=lambda x: x["start"])

    def get_energy_traces(self, service_kind_default="electricity"):
        ''' Retrieve all energy trace records stored as IntervalReading
        elements in the given ESPI Energy Usage XML.

        Energy records are grouped by interpretation and returned in
        EnergyTrace objects.

        Parameters
        ----------
        service_kind_default : str
            Default fuel type to use in parser if ReadingType/commodity field
            is missing.

        Yields
        ------
        energy_trace : eemeter.structures.EnergyTrace
            Energy data traces as described in the xml file.
        '''

        INTERPRETATION_MAPPING = {
            ("electricity", "forward"): "ELECTRICITY_CONSUMPTION_SUPPLIED",
            ("natural_gas", "forward"): "NATURAL_GAS_CONSUMPTION_SUPPLIED",
            ("electricity", "reverse"):
                "ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED",
            ("electriicty", "net"): "ELECTRICITY_CONSUMPTION_NET",
        }

        # Get all consumption records, group by fuel type.
        for flow_direction, records in self._get_consumption_record_groups():

            if len(records) > 0:
                fuel_type_records = defaultdict(list)
                for record in records:
                    fuel_type_records[record["fuel_type"]].append(record)

                # Wrap records in EnergyTrace objects, by fuel type.
                for fuel_type, records in fuel_type_records.items():
                    if fuel_type is None:
                        fuel_type = service_kind_default
                    selector = (fuel_type, flow_direction)
                    interpretation = INTERPRETATION_MAPPING[selector]
                    yield EnergyTrace(
                        interpretation,
                        records=records,
                        unit=records[0]["unit_name"],
                        serializer=ArbitrarySerializer())
