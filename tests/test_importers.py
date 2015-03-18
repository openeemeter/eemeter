import tempfile
import os
from eemeter.importers import import_hpxml

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
