from abc import ABCMeta

class ModelAbstraction:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    ############################################
    #                                          #
    #             ABSTRACT METHODS             #
    #                                          #
    ############################################
    
    @abstractmethod
    def get_method() -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        raise NotImplementedError

    @abstractmethod
    def predict(data) -> str:
        """
        Gets the name of the website from which the data is extracted

        Returns:
            :obj:`str`: Name of the website from which the data is extracted
        """
        raise NotImplementedError