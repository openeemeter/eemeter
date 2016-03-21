from collections import defaultdict
from datetime import datetime, timedelta
from lxml import etree
import os
import pytz

from eemeter.consumption import ConsumptionData

class GreenButtonParser(object):

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
        "61": "VA",
        "38": "W",
        "63": "VAr",
        "71": "VAh",
        "72": "Wh",
        "73": "VArh",
        "29": "V",
        "30": "ohm",
        "5": "A",
        "25": "F",
        "28": "H",
        "23": "degC",
        "27": "sec",
        "159": "min",
        "160": "h",
        "9": "deg",
        "10": "rad",
        "31": "J",
        "32": "n",
        "53": "siemens",
        "0": "none",
        "33": "Hz",
        "3": "g",
        "39": "pa",
        "2": "m",
        "41": "m2",
        "42": "m3",
        "69": "A2",
        "105": "A2h",
        "70": "A2s",
        "106": "Ah",
        "152": "APerA",
        "103": "APerM",
        "68": "As",
        "79": "b",
        "113": "bm",
        "22": "bq",
        "132": "btu",
        "133": "btuPerH",
        "8": "cd",
        "76": "char",
        "75": "HzPerSec",
        "114": "code",
        "65": "cosTheta",
        "111": "count",
        "119": "ft3",
        "120": "ft3compensated",
        "123": "ft3compensatedPerH",
        "78": "gM2",
        "144": "gPerG",
        "21": "gy",
        "150": "HzPerHz",
        "77": "charPerSec",
        "130": "imperialGal",
        "131": "imperialGalPerH",
        "51": "jPerK",
        "165": "jPerKg",
        "6": "K",
        "158": "kat",
        "47": "kgM",
        "48": "kgPerM3",
        "134": "litre",
        "157": "litreCompensated",
        "138": "litreCompensatedPerH",
        "137": "litrePerH",
        "143": "litrePerLitre",
        "82": "litrePerSec",
        "156": "litreUncompensated",
        "139": "litreUncompensatedPerH",
        "35": "lm",
        "34": "lx",
        "49": "m2PerSec",
        "167": "m3compensated",
        "126": "m3compensatedPerH",
        "125": "m3PerH",
        "45": "m3PerSec",
        "166": "m3uncompensated",
        "127": "m3uncompensatedPerH",
        "118": "meCode",
        "7": "mol",
        "147": "molPerKg",
        "145": "molPerM3",
        "146": "molPerMol",
        "80": "money",
        "148": "mPerM",
        "46": "mPerM3",
        "43": "mPerSec",
        "44": "mPerSec2",
        "102": "ohmM",
        "155": "paA",
        "140": "paG",
        "141": "psiA",
        "142": "psiG",
        "100": "q",
        "161": "q45",
        "163": "q45h",
        "162": "q60",
        "164": "q60h",
        "101": "qh",
        "54": "radPerSec",
        "154": "rev",
        "4": "revPerSec",
        "149": "secPerSec",
        "11": "sr",
        "109": "status",
        "24": "sv",
        "37": "t",
        "169": "therm",
        "108": "timeStamp",
        "128": "usGal",
        "129": "usGalPerH",
        "67": "V2",
        "104": "V2h",
        "117": "VAhPerRev",
        "116": "VArhPerRev",
        "74": "VPerHz",
        "151": "VPerV",
        "66": "Vs",
        "36": "wb",
        "107": "WhPerM3",
        "115": "WhPerRev",
        "50": "wPerMK",
        "81": "WPerSec",
        "153": "WPerVA",
        "168": "WPerW",
    }

    VALUE_PARSERS = {'{http://naesb.org/espi}accumulationBehaviour': ACCUMULATION_KIND.get,
                     '{http://naesb.org/espi}commodity': COMMODITY_KIND.get,
                     '{http://naesb.org/espi}dataQualifier': DATA_QUALIFIER_KIND.get,
                     '{http://naesb.org/espi}defaultQuality': QUALITY_OF_READING.get,
                     '{http://naesb.org/espi}flowDirection': FLOW_DIRECTION_KIND.get,
                     '{http://naesb.org/espi}intervalLength': lambda x: timedelta(seconds=int(x)),
                     '{http://naesb.org/espi}kind': MEASUREMENT_KIND.get,
                     '{http://naesb.org/espi}powerOfTenMultiplier': lambda x: int(x),
                     '{http://naesb.org/espi}timeAttribute': TIME_ATTRIBUTE_KIND.get,
                     '{http://naesb.org/espi}uom': UNIT_SYMBOL_KIND.get,
                     '{http://naesb.org/espi}measuringPeriod': TIME_ATTRIBUTE_KIND.get}

    def __init__(self, xml):
        self.root = etree.fromstring(xml)
        self.timezone = self.get_timezone()

    @staticmethod
    def pprint(element):
        print(etree.tostring(element, pretty_print=True))

    def get_usage_point_entry_element(self):
        return self.root.find('.//{http://naesb.org/espi}UsagePoint').getparent().getparent()

    def get_meter_reading_entry_element(self):
        return self.root.find('.//{http://naesb.org/espi}MeterReading').getparent().getparent()

    def get_usage_summary_entry_elements(self):
        usage_summaries = self.root.findall('.//{http://naesb.org/espi}UsageSummary')
        return [e.getparent().getparent() for e in usage_summaries]

    def _normalize_fuel_type(self, uom):
        '''
        Convert ESPI fuel type codes to eemeter fuel type codes.
        '''
        fuel_types = {"naturalGas": "natural_gas",
                      "electricity SecondaryMetered": "electricity"}
        try:
            return fuel_types[uom]
        except KeyError:
            return uom

    def _tz_offset_to_timezone(self, tz_offset):
        '''Convert ESPI timezone offset code to python timezone object.'''
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

    def get_timezone(self):
        '''
        Fetch the timezone the interval readings are in, from
        the ESPI LocalTimeParameters object.
        '''
        local_time_parameters = self.root.find('.//{http://naesb.org/espi}LocalTimeParameters')
        # Parse Daylight Savings Time elements.
        # The start rule and end rule are weird encoded ways of saying when
        # DST should be in effect, and the offset is the actual effect.
        dst_start_rule = local_time_parameters.find('{http://naesb.org/espi}dstStartRule').text
        dst_end_rule = local_time_parameters.find('{http://naesb.org/espi}dstEndRule').text
        dst_offset = local_time_parameters.find('{http://naesb.org/espi}dstOffset').text
        # Check that timezone is a standard timezone.
        # Non-standard timezones might not have DST attributes,
        # break loudly if you encounter them.
        assert dst_start_rule == "360E2000"
        assert dst_end_rule == "B40E2000"
        assert dst_offset == "3600"

        # Find the ESPI timezone offset code, and convert it to
        # a python timezone object.
        tz_offset = local_time_parameters.find('{http://naesb.org/espi}tzOffset').text
        return self._tz_offset_to_timezone(tz_offset)

    class ChildElementGetter(object):
        '''
        Helper class that gets child (or really descendant) elements
        of given element, extract their text values, and parses them.
        '''
        def __init__(self, element, value_parsers):
            self.element = element
            # Different child elements have different value parsing functions.
            self.VALUE_PARSERS = value_parsers

        def child_element_value(self, child_element_name):
            '''Return parsed text value of child element.'''
            child_element = self.element.find(child_element_name)
            if child_element is not None:
                try:
                    return self.VALUE_PARSERS[child_element_name](child_element.text)
                except KeyError:
                    msg = 'No parsing function defined for text value of \
                           element %s' % child_element_name
                    raise NotImplementedError(msg)

    def get_reading_type(self):
        '''
        Get and parse the first ReadingType element. Use to describe all interval readings
        in the XML file.
        '''
        # Grab the first reading element you run into.
        # Note: this assumes that ReadingType is the same for all IntervalBlocks.
        reading_type_element = self.root.findall('.//{http://naesb.org/espi}ReadingType')[0]

        # Initialize Getter class for reading type element, to make getting and parsing
        # the values of child elements easier.
        reading_type = self.ChildElementGetter(reading_type_element, self.VALUE_PARSERS)

        return {'accumulation_behavior': reading_type.child_element_value('{http://naesb.org/espi}accumulationBehaviour'),
                'commodity': reading_type.child_element_value('{http://naesb.org/espi}commodity'),
                'data_qualifier': reading_type.child_element_value('{http://naesb.org/espi}dataQualifier'),
                'default_quality': reading_type.child_element_value('{http://naesb.org/espi}defaultQuality'),
                'flow_direction': reading_type.child_element_value('{http://naesb.org/espi}flowDirection'),
                'interval_length': reading_type.child_element_value('{http://naesb.org/espi}intervalLength'),
                'kind': reading_type.child_element_value('{http://naesb.org/espi}kind'),
                'power_of_ten_multiplier': reading_type.child_element_value('{http://naesb.org/espi}powerOfTenMultiplier'),
                'time_attribute': reading_type.child_element_value('{http://naesb.org/espi}timeAttribute'),
                'uom': reading_type.child_element_value('{http://naesb.org/espi}uom'),
                'measuring_period': reading_type.child_element_value('{http://naesb.org/espi}measuringPeriod')}

    def parse_interval_reading(self, interval_reading):
        '''
        Parse ESPI IntervalReading element into dict.

        IntervalReadings contain the core data observations that
        drive the eemeter: interval energy use measurements.

        This method uses document-level timezone attribute to
        correctly parse interval start times into tz-aware datetimes.
        '''
        reading_quality_element = interval_reading.find("{http://naesb.org/espi}ReadingQuality/{http://naesb.org/espi}quality")
        reading_quality = self.QUALITY_OF_READING[reading_quality_element.text]

        duration_element = interval_reading.find("{http://naesb.org/espi}timePeriod/{http://naesb.org/espi}duration")
        duration = timedelta(seconds=int(duration_element.text))

        start_element = interval_reading.find("{http://naesb.org/espi}timePeriod/{http://naesb.org/espi}start")
        start = datetime.fromtimestamp(int(start_element.text), tz=self.timezone)

        value = int(interval_reading.find("{http://naesb.org/espi}value").text)

        return {"reading_quality": reading_quality,
                "duration": duration,
                "start": start,
                "value": value}

    def parse_interval_block(self, interval_block):
        '''
        Parse ESPI IntervalBlock element - and child IntervalReadings
        elements - into dict.

        IntervalBlocks typically hold 24-hours worth of IntervalReadings.
        In addition interval readings, return the start and duration of the
        block, and a sibling ReadingType element which describes the block's
        readings.
        '''
        # Capture start and duration of the interval block.
        interval_duration_element = interval_block.find("{http://naesb.org/espi}interval/{http://naesb.org/espi}duration")
        interval_start_element = interval_block.find("{http://naesb.org/espi}interval/{http://naesb.org/espi}start")
        interval_duration = timedelta(seconds=int(interval_duration_element.text))
        interval_start = datetime.fromtimestamp(int(interval_start_element.text), tz=self.timezone)

        # Fetch sibling ReadingType for block.
        reading_type = self.get_reading_type()

        # Collect and parse all interval readings for the block.
        interval_readings = [self.parse_interval_reading(reading) for reading
                             in interval_block.findall("{http://naesb.org/espi}IntervalReading")]

        return {"interval": {"duration": interval_duration,
                             "start": interval_start},
                "reading_type": reading_type,
                "interval_readings": interval_readings}

    def get_interval_blocks(self):
        '''
        Return all interval blocks in ESPI Energy Usage XML.
        Each interval block contains a set of interval readings.
        '''
        interval_block_tags = self.root.findall('.//{http://naesb.org/espi}IntervalBlock')
        for interval_block_tag in interval_block_tags:
            yield self.parse_interval_block(interval_block_tag)

    def get_consumption_records(self):
        '''
        Return all consumption records, across all interval blocks,
        stored in ESPI Energy Usage XML.
        '''
        for interval_block in self.get_interval_blocks():
            fuel_type = self._normalize_fuel_type(interval_block["reading_type"]["commodity"])
            # Values must be adjusted with interval-block level multiplier.
            multiplier = 10 ** interval_block["reading_type"]["power_of_ten_multiplier"]
            unit_name = interval_block["reading_type"]["uom"]
            # Package block readings with adjusted units and block fuel type.
            for interval_reading in interval_block["interval_readings"]:
                yield {"start": interval_reading["start"],
                       "end": interval_reading["start"] + interval_reading["duration"],
                       "value": interval_reading["value"] * multiplier,
                       "estimated": "estimated" in interval_reading["reading_quality"],
                       "fuel_type": fuel_type,
                       "unit_name": unit_name}

    def get_consumption_data_objects(self):
        '''
        Retrieve all consumption records stored as Interval Reading elements
        in  the given ESPI Energy Usage XML.

        Consumption records are grouped by fuel type and returned in
        ConsumptionData objects.
        '''
        # Get all consumption records, group by fuel type.
        fuel_type_records = defaultdict(list)
        for record in self.get_consumption_records():
            fuel_type_records[record["fuel_type"]].append(record)
        # Wrap records in ConsumptionData objects.
        for fuel_type, records in fuel_type_records.items():
            yield ConsumptionData(records, fuel_type,
                                  records[0]["unit_name"],
                                  record_type='arbitrary')
