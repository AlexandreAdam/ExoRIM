from abc import abstractmethod

# Abstraction of important methods in the code (Methods that are used throughout the project are abstracted here)

#TODO check how to do this
class BaselinesBase:
    pass
    # @property
    # def BLM(self):
    #     return type(self).BLM
    #
    # @property
    # def UVC(self):
    #     return  type(self).UVC


class PhysicalModelBase:
    pass
#     @abstractmethod
#     def log_likelihood(self, image, X):
#         raise NotImplementedError(f"{self.__name__} do not implement log_likelihood method")
#
#     @abstractmethod
#     def grad_log_likelihood(self, image, X):
#         raise NotImplementedError()
#
#     @abstractmethod
#     def bispectrum(self, X):
#         raise NotImplementedError()
#
#     @property
#     def CPO(self):
#         return type(self).CPO
#
#     @property
#     def A(self):
#         return type(self).A
#
#     @property
#     def A1(self):
#         return type(self).A1
#
#     @property
#     def A2(self):
#         return type(self).A2
#
#     @property
#     def A3(self):
#         return type(self).A3
