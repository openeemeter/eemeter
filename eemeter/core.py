from datetime import date, timedelta
import warnings

class EnergyBill:

    def __init__(self,usage,start_date,end_date,estimated=False):
        self.usage = usage
        self.start_date = start_date
        self.end_date = end_date
        self.estimated = estimated

    def __str__(self):
        return "Energy Bill {}: {}".format(self.end_date,self.usage)

    def days(self):
        """
        Return the number of days this bill covers.
        """
        return (self.end_date - self.start_date).days + 1

    def is_valid(self):
        """
        Return true if the bill is has a reasonable date range.
        """
        if not isinstance(self.start_date,date):
            return False
        if not isinstance(self.end_date,date):
            return False
        valid_date_range = self.days() >= 1
        return valid_date_range

class Building:

    def __init__(self,energy_bills):
        self.energy_bills = energy_bills

    def __str__(self):
        return "Building ({} bills)".format(len(self.energy_bills))

    def meets_calibration_criteria(self):
        """
        Return true if the energy bills associated with the building meet
        detailed calibration criteria specified in the
        ANSI/BPI-2400-S-2012 standard.
        """
        most_recent_bill = self.most_recent_energy_bill()
        if most_recent_bill is None:
            return False
        if most_recent_bill.end_date <= date.today() - timedelta(days=365):
            return False
        if most_recent_bill.end_date > date.today():
            return False
        return True

    def most_recent_energy_bill(self):
        """
        Return the most recent energy bill, or None if none is available.
        """
        if self.energy_bills == []:
            return None
        return max(self.energy_bills,key=lambda x: x.end_date)

    def total_usage(self):
        """
        Return total usage of all energy bills.
        """
        return sum([bill.usage for bill in self.energy_bills])

    def consolidate_estimated_reads(self):
        """
        Consolidate estimated reads with actual reads using the procedure
        outlined in the ANSI/BPI-2400-S-2012 standard.
        """
        sorted_bills = sorted(self.energy_bills,key=lambda x: x.end_date)
        if sorted_bills is []:
            warnings.warn("No bills to consolidate")
            return
        new_bills = [sorted_bills[0]]
        for bill in sorted_bills[1:-1]:
            if bill.estimated:
                new_bills[-1].end_date = bill.end_date
                new_bills[-1].usage += bill.usage
            else:
                new_bills.append(bill)
        if not sorted_bills[-1].estimated:
            new_bills.append(sorted_bills[-1])
        self.energy_bills = new_bills

    def no_missing_days(self):
        """
        Return true if there are no missing days in the energy bills.
        """
        sorted_bills = sorted(self.energy_bills,key=lambda x: x.end_date)
        for prev_bill,next_bill in zip(sorted_bills,sorted_bills[1:]):
            if not (next_bill.start_date - prev_bill.end_date).days == 1:
                return False
        return True
