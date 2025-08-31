from project.shared.meta.singleton import SingletonMeta
from project.observers.observable import Observable


class ObservableSingletonMeta(SingletonMeta, type(Observable)):
    pass
