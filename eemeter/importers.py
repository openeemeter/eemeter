from lxml import etree
from dateutil.parser import parse
from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory
from datetime import datetime

hpxml_fuel_type_mapping = {
    "electricity": "electricity",
    "natural gas": "natural_gas",
}

def import_hpxml(filename):
    with(open(filename,'r')) as f:
        tree = etree.parse(f)

    root = tree.getroot()
    ns = root.nsmap

    consumption_info_xpath = "ns2:Consumption/ns2:ConsumptionDetails/ns2:ConsumptionInfo"
    consumption_infos = root.xpath(consumption_info_xpath,namespaces=ns)

    consumptions = []
    for info in consumption_infos:
        fuel_type = info.xpath("ns2:ConsumptionType/ns2:Energy/ns2:FuelType",namespaces=ns)
        unit_of_measure = info.xpath("ns2:ConsumptionType/ns2:Energy/ns2:UnitofMeasure",namespaces=ns)
        if fuel_type == [] or unit_of_measure == []:
            continue
        fuel_type = hpxml_fuel_type_mapping[fuel_type[0].text]
        unit_str = unit_of_measure[0].text
        consumption_details = info.xpath("ns2:ConsumptionDetail",namespaces=ns)
        for details in consumption_details:
            usage = float(details.xpath("ns2:Consumption",namespaces=ns)[0].text)
            start = parse(details.xpath("ns2:StartDateTime",namespaces=ns)[0].text)
            end = parse(details.xpath("ns2:EndDateTime",namespaces=ns)[0].text)
            reading_type = details.xpath("ns2:ReadingType",namespaces=ns)
            if reading_type == []:
                estimated = False
            else:
                estimated = (reading_type[0].text == "estimated")
            consumption = Consumption(usage,unit_str,fuel_type,start,end,estimated)
            consumptions.append(consumption)
    return ConsumptionHistory(consumptions)

def import_green_button_xml(filename):
    with(open(filename,'r')) as f:
        tree = etree.parse(f)

    interval_blocks = tree.xpath("//*[local-name() = 'IntervalBlock']")

    def get_consumption(reading):
        usage = int(reading.xpath("*[local-name() = 'value']")[0].text)
        fuel_type = "electricity"
        unit_str = "Wh"
        time_period = reading.xpath("*[local-name() = 'timePeriod']")[0]
        start_ms = int(time_period.xpath("*[local-name() = 'start']")[0].text)
        duration_ms = int(time_period.xpath("*[local-name() = 'duration']")[0].text)
        end_ms = start_ms + duration_ms
        start = datetime.fromtimestamp(start_ms)
        end = datetime.fromtimestamp(end_ms)
        return Consumption(usage,unit_str,fuel_type,start,end)

    consumptions = []
    for block in interval_blocks:
        readings = block.xpath("*[local-name() = 'IntervalReading']")
        for reading in readings:
            consumption = get_consumption(reading)
            consumptions.append(consumption)

    return ConsumptionHistory(consumptions)
