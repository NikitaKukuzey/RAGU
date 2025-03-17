from typing import ClassVar


class Registrable:
    """
    A base class for registering subclasses dynamically and retrieving them by name.

    This class manages a registry of subclasses that can be registered and retrieved using 
    their names. The `register` method registers a subclass, and the `get` method retrieves 
    a registered subclass.

    :param name: The name used to register the subclass.
    :param class_name: The name of the class to retrieve from the registry.
    :param *args: Arguments passed to the constructor of the registered class.
    :param **kwargs: Keyword arguments passed to the constructor of the registered class.
    :return: A subclass of the `Registrable` class, instantiated with the provided arguments.
    """
    _registry: ClassVar[dict] = {}

    @classmethod
    def register(cls, name: str):
        """
        Registers a subclass with a given name in the class's registry.

        :param name: The name under which the subclass will be registered.
        :return: The decorator that registers the subclass.
        """
        def wrapper(subclass):
            if cls not in cls._registry:
                cls._registry[cls] = {}
            registry = cls._registry[cls]
            registry[name] = subclass
            return subclass
        return wrapper

    @classmethod
    def get(cls, class_name: str, *args, **kwargs):
        """
        Retrieves a registered subclass by name and instantiates it.

        If the class is not found in the registry, a ValueError is raised.

        :param class_name: The name of the class to retrieve from the registry.
        :param *args: Arguments passed to the constructor of the subclass.
        :param **kwargs: Keyword arguments passed to the constructor of the subclass.
        :return: An instance of the registered subclass.
        :raises ValueError: If the class with the given name is not found in the registry.
        """
        registry = cls._registry.get(cls, {})
        if class_name not in registry:
            raise ValueError(f"Class named '{class_name}' not found.")
        kwargs.pop('class_name', None)
        return registry[class_name](*args, **kwargs)
    


