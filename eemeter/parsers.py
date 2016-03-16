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

    def __init__(self, xml):
        self.root = etree.fromstring(xml)

    @staticmethod
    def pprint(element):
        print(etree.tostring(element, pretty_print=True))

    def get_id_element(self):
        return self.root.find('ns1:id', namespaces=self.root.nsmap)

    def get_title_element(self):
        return self.root.find('ns1:title', namespaces=self.root.nsmap)

    def get_updated_element(self):
        return self.root.find('ns1:updated', namespaces=self.root.nsmap)

    def get_link_element(self):
        return self.root.find('ns1:link', namespaces=self.root.nsmap)

    def get_entry_elements(self):
        return self.root.findall('ns1:entry', namespaces=self.root.nsmap)

    def get_local_time_parameters_entry_element(self):
        return self.root.find('.//ns0:LocalTimeParameters', namespaces=self.root.nsmap).getparent().getparent()

    def parse_local_time_parameters_entry(self, entry):
        local_time_parameters = entry.find('ns1:content/ns0:LocalTimeParameters', namespaces=entry.nsmap)

        dst_start_rule = local_time_parameters.find('ns0:dstStartRule', namespaces=local_time_parameters.nsmap).text
        dst_end_rule = local_time_parameters.find('ns0:dstEndRule', namespaces=local_time_parameters.nsmap).text
        dst_offset = local_time_parameters.find('ns0:dstOffset', namespaces=local_time_parameters.nsmap).text
        assert dst_start_rule == "360E2000"
        assert dst_end_rule == "B40E2000"
        assert dst_offset == "3600"

        tz_offset = local_time_parameters.find('ns0:tzOffset', namespaces=local_time_parameters.nsmap).text

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

    def get_reading_type_entry_elements(self):
        reading_types = self.root.findall('.//ns0:ReadingType', namespaces=self.root.nsmap)
        return [e.getparent().getparent() for e in reading_types]

    def parse_reading_type_entry(self, entry):
        reading_type = entry.find('ns1:content/ns0:ReadingType', namespaces=entry.nsmap)

        accumulation_behavior_element = reading_type.find('ns0:accumulationBehaviour', namespaces=reading_type.nsmap)
        if accumulation_behavior_element is not None:
            accumulation_behavior = self.ACCUMULATION_KIND.get(accumulation_behavior_element.text)
        else:
            accumulation_behavior = None

        commodity_element = reading_type.find('ns0:commodity', namespaces=reading_type.nsmap)
        if commodity_element is not None:
            commodity = self.COMMODITY_KIND.get(commodity_element.text)
        else:
            commodity = None

        data_qualifier_element = reading_type.find('ns0:dataQualifier', namespaces=reading_type.nsmap)
        if data_qualifier_element is not None:
            data_qualifier = self.DATA_QUALIFIER_KIND.get(data_qualifier_element.text)
        else:
            data_qualifier = None

        default_quality_element = reading_type.find('ns0:defaultQuality', namespaces=reading_type.nsmap)
        if default_quality_element is not None:
            default_quality = self.QUALITY_OF_READING.get(default_quality_element.text)
        else:
            default_quality = None

        flow_direction_element = reading_type.find('ns0:flowDirection', namespaces=reading_type.nsmap)
        if flow_direction_element is not None:
            flow_direction = self.FLOW_DIRECTION_KIND.get(flow_direction_element.text)
        else:
            flow_direction = None

        interval_length_element = reading_type.find('ns0:intervalLength', namespaces=reading_type.nsmap)
        if interval_length_element is not None:
            interval_length = timedelta(seconds=int(interval_length_element.text))
        else:
            interval_length = None

        kind_element = reading_type.find('ns0:kind', namespaces=reading_type.nsmap)
        if kind_element is not None:
            kind = self.MEASUREMENT_KIND.get(kind_element.text)
        else:
            kind = None

        power_of_ten_multiplier_element = reading_type.find('ns0:powerOfTenMultiplier', namespaces=reading_type.nsmap)
        if power_of_ten_multiplier_element is not None:
            power_of_ten_multiplier = int(power_of_ten_multiplier_element.text)
        else:
            power_of_ten_multiplier = None

        time_attribute_element = reading_type.find('ns0:timeAttribute', namespaces=reading_type.nsmap)
        if time_attribute_element is not None:
            time_attribute = self.TIME_ATTRIBUTE_KIND.get(time_attribute_element.text)
        else:
            time_attribute = None

        uom_element = reading_type.find('ns0:uom', namespaces=reading_type.nsmap)
        if uom_element is not None:
            uom = self.UNIT_SYMBOL_KIND.get(uom_element.text)
        else:
            uom = None

        measuring_period_element = reading_type.find('ns0:measuringPeriod', namespaces=reading_type.nsmap)
        if measuring_period_element is not None:
            measuring_period = self.TIME_ATTRIBUTE_KIND.get(measuring_period_element.text)
        else:
            measuring_period = None

        data = {
            "accumulation_behavior": accumulation_behavior,
            "commodity": commodity,
            "data_qualifier": data_qualifier,
            "default_quality": default_quality,
            "flow_direction": flow_direction,
            "interval_length": interval_length,
            "kind": kind,
            "power_of_ten_multiplier": power_of_ten_multiplier,
            "time_attribute": time_attribute,
            "uom": uom,
            "measuring_period": measuring_period,
        }
        return data

    def get_usage_point_entry_element(self):
        return self.root.find('.//ns0:UsagePoint', namespaces=self.root.nsmap).getparent().getparent()

    def get_meter_reading_entry_element(self):
        return self.root.find('.//ns0:MeterReading', namespaces=self.root.nsmap).getparent().getparent()

    def get_interval_block_entry_elements(self):
        interval_blocks = self.root.findall('.//ns0:IntervalBlock', namespaces=self.root.nsmap)
        return [e.getparent().getparent() for e in interval_blocks]

    def get_usage_summary_entry_elements(self):
        usage_summaries = self.root.findall('.//ns0:UsageSummary', namespaces=self.root.nsmap)
        return [e.getparent().getparent() for e in usage_summaries]

    def parse_interval_block_entries(self, entries):
        data = []
        local_time_parameters_entry = self.get_local_time_parameters_entry_element()
        timezone = self.parse_local_time_parameters_entry(local_time_parameters_entry)
        for entry in entries:
            interval_block = entry.find('ns1:content/ns0:IntervalBlock', namespaces=entry.nsmap)
            parsed = self.parse_interval_block(interval_block, timezone)
            data.append(parsed)
        return data

    def parse_interval_block(self, interval_block, timezone):

        interval_duration_element = interval_block.find("ns0:interval/ns0:duration", namespaces=interval_block.nsmap)
        interval_start_element = interval_block.find("ns0:interval/ns0:start", namespaces=interval_block.nsmap)
        interval_duration = timedelta(seconds=int(interval_duration_element.text))
        interval_start = datetime.fromtimestamp(int(interval_start_element.text), tz=timezone)

        # grab first reading_type
        reading_type_element = parser.get_reading_type_entry_elements()[0]
        reading_type = self.parse_reading_type_entry(reading_type_element)


        interval_readings = []
        for interval_reading in interval_block.findall("ns0:IntervalReading", namespaces=interval_block.nsmap):
            interval_readings.append(self.parse_interval_reading(interval_reading, timezone))

        data = {
            "reading_type": reading_type,
            "interval": {
                "duration": interval_duration,
                "start": interval_start,
            },
            "interval_readings": interval_readings,
        }
        return data

    def parse_interval_reading(self, interval_reading, timezone):
        reading_quality_element = interval_reading.find("ns0:ReadingQuality/ns0:quality", namespaces=interval_reading.nsmap)
        reading_quality = self.QUALITY_OF_READING[reading_quality_element.text]
        duration_element = interval_reading.find("ns0:timePeriod/ns0:duration", namespaces=interval_reading.nsmap)
        start_element = interval_reading.find("ns0:timePeriod/ns0:start", namespaces=interval_reading.nsmap)
        duration = timedelta(seconds=int(duration_element.text))
        start = datetime.fromtimestamp(int(start_element.text), tz=timezone)
        value = int(interval_reading.find("ns0:value", namespaces=interval_reading.nsmap).text)
        data = {
            "reading_quality": reading_quality,
            "duration": duration,
            "start": start,
            "value": value,
        }
        return data

    def get_interval_block_data(self):
        entries = parser.get_interval_block_entry_elements()
        data = parser.parse_interval_block_entries(entries)
        return data

    def get_consumption_records(self):
        data = self.get_interval_block_data()
        records = []
        for interval_block in data:
            multiplier = 10 ** interval_block["reading_type"]["power_of_ten_multiplier"]
            fuel_type = self._normalize_fuel_type(interval_block["reading_type"]["commodity"])
            unit_name = interval_block["reading_type"]["uom"]
            for interval_reading in interval_block["interval_readings"]:
                record = {
                    "start": interval_reading["start"],
                    "end": interval_reading["start"] + interval_reading["duration"],
                    "value": interval_reading["value"] * multiplier,
                    "estimated": "estimated" in interval_reading["reading_quality"],
                    "fuel_type": fuel_type,
                    "unit_name": unit_name,
                }
                records.append(record)
        return records

    def get_consumption_data_objects(self):
        records = self.get_consumption_records()
        fuel_type_records = defaultdict(list)
        for record in records:
            fuel_type_records[record["fuel_type"]].append(record)
        cds = []
        for fuel_type, records in fuel_type_records.items():
            cd = ConsumptionData(records, fuel_type, records[0]["unit_name"], record_type='arbitrary')
            cds.append(cd)
        return cds

    def _normalize_fuel_type(self, uom):
        fuel_types = {
            "naturalGas": "natural_gas",
            "electricity SecondaryMetered": "electricity",
        }

        try:
            return fuel_types[uom]
        except KeyError:
            return uom
