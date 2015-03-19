import tempfile
import os
from eemeter.importers import import_hpxml
from eemeter.importers import import_green_button_xml

from numpy.testing import assert_allclose

RTOL = 1e-2
ATOL = 1e-2

def test_import_hpxml():
    data0 = """<?xml version="1.0" encoding="UTF-8"?>
    <ns2:HPXML xmlns:ns2="http://hpxmlonline.com/2014/6" schemaVersion="2.0">
      <ns2:XMLTransactionHeaderInformation />
      <ns2:SoftwareInfo />
      <ns2:Contractor />
      <ns2:Customer />
      <ns2:Building />
      <ns2:Project />
      <ns2:Utility />
      <ns2:Consumption>
        <ns2:BuildingID />
        <ns2:CustomerID />
        <ns2:ConsumptionDetails>
          <ns2:ConsumptionInfo>
            <ns2:UtilityID />
            <ns2:ConsumptionType>
              <ns2:Energy>
                <ns2:FuelType>electricity</ns2:FuelType>
                <ns2:UnitofMeasure>kWh</ns2:UnitofMeasure>
              </ns2:Energy>
            </ns2:ConsumptionType>
            <ns2:ConsumptionDetail>
              <ns2:Consumption>10.0</ns2:Consumption>
              <ns2:StartDateTime>2014-01-01T00:00:00+00:00</ns2:StartDateTime>
              <ns2:EndDateTime>2014-02-01T00:00:00+00:00</ns2:EndDateTime>
              <ns2:ReadingType>total</ns2:ReadingType>
            </ns2:ConsumptionDetail>
            <ns2:ConsumptionDetail>
              <ns2:Consumption>11.0</ns2:Consumption>
              <ns2:StartDateTime>2014-02-01T00:00:00+00:00</ns2:StartDateTime>
              <ns2:EndDateTime>2014-02-28T00:00:00+00:00</ns2:EndDateTime>
              <ns2:ReadingType>total</ns2:ReadingType>
            </ns2:ConsumptionDetail>
            <ns2:ConsumptionDetail>
              <ns2:Consumption>12.0</ns2:Consumption>
              <ns2:StartDateTime>2014-02-28T00:00:00+00:00</ns2:StartDateTime>
              <ns2:EndDateTime>2014-03-29T00:00:00+00:00</ns2:EndDateTime>
              <ns2:ReadingType>estimated</ns2:ReadingType>
            </ns2:ConsumptionDetail>
          </ns2:ConsumptionInfo>
          <ns2:ConsumptionInfo>
            <ns2:UtilityID />
            <ns2:ConsumptionType>
              <ns2:Energy>
                <ns2:FuelType>natural gas</ns2:FuelType>
                <ns2:UnitofMeasure>therms</ns2:UnitofMeasure>
              </ns2:Energy>
            </ns2:ConsumptionType>
            <ns2:ConsumptionDetail>
              <ns2:Consumption>10.0</ns2:Consumption>
              <ns2:StartDateTime>2014-01-01T00:00:00+00:00</ns2:StartDateTime>
              <ns2:EndDateTime>2014-02-01T00:00:00+00:00</ns2:EndDateTime>
              <ns2:ReadingType>total</ns2:ReadingType>
            </ns2:ConsumptionDetail>
            <ns2:ConsumptionDetail>
              <ns2:Consumption>11.0</ns2:Consumption>
              <ns2:StartDateTime>2014-02-01T00:00:00+00:00</ns2:StartDateTime>
              <ns2:EndDateTime>2014-02-28T00:00:00+00:00</ns2:EndDateTime>
              <ns2:ReadingType>total</ns2:ReadingType>
            </ns2:ConsumptionDetail>
            <ns2:ConsumptionDetail>
              <ns2:Consumption>12.0</ns2:Consumption>
              <ns2:StartDateTime>2014-02-28T00:00:00+00:00</ns2:StartDateTime>
              <ns2:EndDateTime>2014-03-29T00:00:00+00:00</ns2:EndDateTime>
              <ns2:ReadingType>estimated</ns2:ReadingType>
            </ns2:ConsumptionDetail>
          </ns2:ConsumptionInfo>
        </ns2:ConsumptionDetails>
      </ns2:Consumption>
    </ns2:HPXML>
    """
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(data0.encode('utf-8'))

    ch = import_hpxml(fname)
    consumptions_e = ch.electricity
    consumptions_g = ch.natural_gas
    assert_allclose([ c.kWh for c in consumptions_e],[10.,11.,12.],rtol=RTOL,atol=ATOL)
    assert_allclose([ c.therms for c in consumptions_g],[10.,11.,12.],rtol=RTOL,atol=ATOL)
    assert len(consumptions_e) == 3
    assert len(consumptions_g) == 3
    assert not consumptions_e[0].estimated
    assert not consumptions_e[1].estimated
    assert not consumptions_g[0].estimated
    assert not consumptions_g[1].estimated
    assert consumptions_e[2].estimated
    assert consumptions_g[2].estimated

def test_import_green_button_xml():
    data0 = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:espi="http://naesb.org/espi" xsi:schemaLocation="http://naesb.org/espi espiDerived.xsd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <id>urn:uuid:1DA5F5DD-312B-4896-B910-098F59C73965</id>
        <title>Green Button Subscription Feed</title>
        <updated>2013-09-19T04:00:00Z</updated>
        <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/Subscription/83e269c1"/>
        <entry>
        <id>urn:uuid:C8C34B3A-D175-447B-BD00-176F60194DE0</id>
            <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1"/>
            <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint"/>
            <link rel="related" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading"/>
            <link rel="related" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/ElectricPowerUsageSummary"/>
            <link rel="related" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/LocalTimeParameters/01"/>
            <title>Green Button Sample Data File</title>
            <content>
                <UsagePoint xmlns="http://naesb.org/espi">
                    <ServiceCategory>
                        <kind>0</kind>
                    </ServiceCategory>
                </UsagePoint>
            </content>
            <published>2013-09-19T04:00:00Z</published>
            <updated>2013-09-19T04:00:00Z</updated>
        </entry>
        <entry>
        <id>urn:uuid:E30CE77D-EC22-4DA5-83C2-991BA34C97D6</id>
                <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/LocalTimeParameters/01"/>
                <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/LocalTimeParameters"/>
                <title>DST For North America</title>
                <content>
                <LocalTimeParameters xmlns="http://naesb.org/espi">
                            <dstEndRule>B40E2000</dstEndRule>
                            <dstOffset>3600</dstOffset>
                            <dstStartRule>360E2000</dstStartRule>
                            <tzOffset>-18000</tzOffset>
                </LocalTimeParameters>
            </content>
            <published>2013-09-19T04:00:00Z</published>
            <updated>2013-09-19T04:00:00Z</updated>
        </entry>
        <entry>
        <id>urn:uuid:4234AE39-FB6D-48CA-8856-AC9F41FB3D34</id>
            <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading/01"/>
            <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading"/>
            <link rel="related" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading/01/IntervalBlock"/>
            <link rel="related" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/ReadingType/3"/>
            <title>Monthly Electricity Consumption</title>
            <content>
                <MeterReading xmlns="http://naesb.org/espi"/>
            </content>
            <published>2013-09-19T04:00:00Z</published>
            <updated>2013-09-19T04:00:00Z</updated>
        </entry>
        <entry>
        <id>urn:uuid:99B292FC-55F7-4F27-A3B9-CDDAB97CCA90</id>
            <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/ReadingType/3"/>
            <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/ReadingType"/>
            <title>Type of Meter Reading Data</title>
            <content>
                <ReadingType xmlns="http://naesb.org/espi">
                    <accumulationBehaviour>4</accumulationBehaviour>
                    <commodity>1</commodity>
                    <currency>840</currency>
                    <dataQualifier>12</dataQualifier>
                    <flowDirection>1</flowDirection>
                    <intervalLength>86400</intervalLength>
                    <kind>12</kind>
                    <phase>769</phase>
                    <powerOfTenMultiplier>0</powerOfTenMultiplier>
                    <timeAttribute>0</timeAttribute>
                    <uom>72</uom>
                </ReadingType>
            </content>
            <published>2013-09-19T04:00:00Z</published>
            <updated>2013-09-19T04:00:00Z</updated>
        </entry>
        <entry>
            <id>urn:uuid:83D07D60-5A94-4161-A558-A0E6549A0AF2</id>
            <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading/01/IntervalBlock/C"/>
            <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading/01/IntervalBlock"/>
            <title/>
            <content>
                <IntervalBlock xmlns="http://naesb.org/espi">
                    <interval>
                        <duration>2678400</duration>
                        <start>1385874000</start>
                        <!-- start date: 12/1/2013 5:00:00 AM -->
                    </interval>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1385874000</start>
                             <!-- 12/1/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1385960400</start>
                             <!-- 12/2/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386046800</start>
                             <!-- 12/3/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386133200</start>
                             <!-- 12/4/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386219600</start>
                             <!-- 12/5/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386306000</start>
                             <!-- 12/6/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386392400</start>
                             <!-- 12/7/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386478800</start>
                             <!-- 12/8/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386565200</start>
                             <!-- 12/9/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386651600</start>
                             <!-- 12/10/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386738000</start>
                             <!-- 12/11/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386824400</start>
                             <!-- 12/12/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386910800</start>
                             <!-- 12/13/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1386997200</start>
                             <!-- 12/14/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387083600</start>
                             <!-- 12/15/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387170000</start>
                             <!-- 12/16/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387256400</start>
                             <!-- 12/17/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387342800</start>
                             <!-- 12/18/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387429200</start>
                             <!-- 12/19/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387515600</start>
                             <!-- 12/20/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387602000</start>
                             <!-- 12/21/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387688400</start>
                             <!-- 12/22/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387774800</start>
                             <!-- 12/23/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387861200</start>
                             <!-- 12/24/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1387947600</start>
                             <!-- 12/25/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1388034000</start>
                             <!-- 12/26/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1388120400</start>
                             <!-- 12/27/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1388206800</start>
                             <!-- 12/28/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1388293200</start>
                             <!-- 12/29/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1388379600</start>
                             <!-- 12/30/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1388466000</start>
                             <!-- 12/31/2013 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                </IntervalBlock>
            </content>
            <published>2014-01-01T05:00:00Z</published>
            <updated>2014-01-01T05:00:00Z</updated>
        </entry>
        <entry>
            <id>urn:uuid:71B708EC-7181-426B-AC3A-D0F3B49169D9</id>
            <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading/01/IntervalBlock/F"/>
            <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/MeterReading/01/IntervalBlock"/>
            <title/>
            <content>
                <IntervalBlock xmlns="http://naesb.org/espi">
                    <interval>
                        <duration>1724400</duration>
                        <start>1393650000</start>
                        <!-- start date: 3/1/2014 5:00:00 AM -->
                    </interval>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1393650000</start>
                             <!-- 3/1/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1393736400</start>
                             <!-- 3/2/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1393822800</start>
                             <!-- 3/3/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1393909200</start>
                             <!-- 3/4/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1393995600</start>
                             <!-- 3/5/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394082000</start>
                             <!-- 3/6/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394168400</start>
                             <!-- 3/7/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394254800</start>
                             <!-- 3/8/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203112</cost>
                        <timePeriod>
                            <duration>82800</duration>
                            <start>1394341200</start>
                             <!-- 3/9/2014 5:00:00 AM  -->
                        </timePeriod>
                        <value>25389</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394424000</start>
                             <!-- 3/10/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394510400</start>
                             <!-- 3/11/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394596800</start>
                             <!-- 3/12/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394683200</start>
                             <!-- 3/13/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394769600</start>
                             <!-- 3/14/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394856000</start>
                             <!-- 3/15/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>203931</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1394942400</start>
                             <!-- 3/16/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>25662</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1395028800</start>
                             <!-- 3/17/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1395115200</start>
                             <!-- 3/18/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1395201600</start>
                             <!-- 3/19/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                    <IntervalReading>
                        <cost>256347</cost>
                        <timePeriod>
                            <duration>86400</duration>
                            <start>1395288000</start>
                             <!-- 3/20/2014 4:00:00 AM  -->
                        </timePeriod>
                        <value>21021</value>
                    </IntervalReading>
                </IntervalBlock>
            </content>
            <published>2014-03-21T04:00:00Z</published>
            <updated>2014-03-21T04:00:00Z</updated>
        </entry>
        <entry>
            <id>urn:uuid:923A7143-263E-421B-BEA2-E41B7E240013</id>
            <link rel="self" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/ElectricPowerUsageSummary/01"/>
            <link rel="up" href="https://services.greenbuttondata.org/DataCustodian/espi/1_1/resource/RetailCustomer/1/UsagePoint/1/ElectricPowerUsageSummary"/>
            <title>Usage Summary</title>
            <content>
                <ElectricPowerUsageSummary xmlns="http://naesb.org/espi">
                    <billingPeriod>
                        <duration>2419200</duration>
                        <start>1391230800</start>
                    </billingPeriod>
                    <billLastPeriod>6752000</billLastPeriod>
                    <billToDate>4807000</billToDate>
                    <costAdditionalLastPeriod>0</costAdditionalLastPeriod>
                    <currency>840</currency>
                    <overallConsumptionLastPeriod>
                        <powerOfTenMultiplier>0</powerOfTenMultiplier>
                        <uom>72</uom>
                        <value>625716</value>
                    </overallConsumptionLastPeriod>
                    <currentBillingPeriodOverAllConsumption>
                        <powerOfTenMultiplier>0</powerOfTenMultiplier>
                        <timeStamp>1395374400</timeStamp>
                        <uom>72</uom>
                        <value>447993</value>
                    </currentBillingPeriodOverAllConsumption>
                    <qualityOfReading>14</qualityOfReading>
                    <statusTimeStamp>1395374400</statusTimeStamp>
                </ElectricPowerUsageSummary>
            </content>
            <published>2014-03-01T05:00:00Z</published>
            <updated>2014-03-01T05:00:00Z</updated>
        </entry>
    </feed>
    """
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(data0.encode('utf-8'))

    ch = import_green_button_xml(fname)

    consumptions_e = ch.electricity

    c_kWh = [c.kWh for c in consumptions_e]

    assert_allclose(c_kWh, [25.662, 21.021, 21.021, 21.021, 21.021,
                            21.021, 25.662, 25.662, 21.021, 21.021,
                            21.021, 21.021, 21.021, 25.662, 25.662,
                            21.021, 21.021, 21.021, 21.021, 21.021,
                            25.662, 25.662, 21.021, 21.021, 21.021,
                            21.021, 21.021, 25.662, 25.662, 21.021,
                            21.021, 25.662, 25.662, 21.021, 21.021,
                            21.021, 21.021, 21.021, 25.662, 25.389,
                            21.021, 21.021, 21.021, 21.021, 21.021,
                            25.662, 25.662, 21.021, 21.021, 21.021,
                            21.021], rtol=RTOL, atol=ATOL)
