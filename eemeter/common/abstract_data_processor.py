from abc import ABC, abstractmethod

class AbstractDataProcessor(ABC):
    """Abstract class for data processors."""

    @abstractmethod
    def set_data(self, data):
        """Process data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to process.

        Returns
        -------
        processed_data : pd.DataFrame
            Processed data.
        """
        pass



    @abstractmethod
    def _check_data_sufficiency(self, data):
        """Check data sufficiency.

        Parameters
        ----------
        data : pd.DataFrame
            Data to check.

        Returns
        -------
        is_sufficient : bool
            Whether the data is sufficient.
        """
        pass

    @abstractmethod
    def extend(self, data):
        """Extend data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to extend.

        Returns
        -------
        extended_data : pd.DataFrame
            Extended data.
        """
        pass

    @abstractmethod
    def _interpolate_data(self, data):
        pass
    