import abc

class AbstractTransform(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x):
        pass

    def __repr__(self):
        return self.__class__.__name__ + '()'

