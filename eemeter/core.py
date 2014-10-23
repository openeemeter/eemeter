class EnergyBill:

    def __init__(self,usage,start_date,end_date):
        self.usage = usage
        self.start_date = start_date
        self.end_date = end_date

    def days(self):
        assert self.start_date <= self.end_date
        return (self.end_date - self.start_date).days + 1

