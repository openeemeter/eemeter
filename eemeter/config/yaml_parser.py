import yaml
from collections import namedtuple

is_initialized = False

BaseProxy = namedtuple('BaseProxy', ['callable', 'keywords', 'yaml_src'])
Setting = namedtuple('Setting', ['name'])

class Proxy(BaseProxy):

    __slots__ = []

    def __hash__(self):
        """Return a hash based on the object ID (to avoid hashing unhashable
        namedtuple elements).
        """
        return hash(id(self))

def load(stream, settings={}):
    """Load a Meter specification written in YAML format. `stream` can be a
    string or a file object.
    """
    global is_initialized
    if not is_initialized:
        initialize()

    if isinstance(stream, str):
        string = stream
    else:
        string = stream.read()

    proxy_graph = yaml.load(string)
    return _instantiate(proxy_graph, bindings={}, settings=settings)

def load_path(path, settings={}):
    """Load a Meter specification written in YAML format. `path` is the path
    to the yaml file.
    """
    with open(path, 'r') as f:
        content = ''.join(f.readlines())
    return load(content, settings=settings)

def _instantiate_proxy_tuple(proxy, bindings, settings):

    if proxy in bindings:
        return bindings[proxy]
    else:
        kwargs = {k: _instantiate(v, bindings, settings)
                  for k, v in proxy.keywords.items()}
        obj = proxy.callable(**kwargs)

    bindings[proxy] = obj
    return bindings[proxy]

def _instantiate(proxy, bindings={}, settings={}):

    if isinstance(proxy, Proxy):
        instance = _instantiate_proxy_tuple(proxy, bindings, settings)
        if isinstance(instance, Setting):
            if instance.name in settings:
                return settings[instance.name]
            else:
                message = "{} not found in settings dict.".format(instance.name)
                raise KeyError(message)
        else:
            return instance
    elif isinstance(proxy, dict):
        return {k: _instantiate(v , bindings, settings) for k, v in proxy.items()}
    elif isinstance(proxy, list):
        return [_instantiate(v, bindings, settings) for v in proxy]
    else:
        return proxy

def try_to_import(tag_suffix):
    components = tag_suffix.split('.')
    modulename = '.'.join(components[:-1])

    exec('import %s' % modulename)
    obj = eval(tag_suffix)
    return obj

def initialize():
    """Add constructors to yaml parser.
    """
    from eemeter.meter.base import MeterBase
    from eemeter.models.temperature_sensitivity import Model

    yaml.add_multi_constructor('!obj:', multi_constructor_obj)
    yaml.add_constructor('!setting', constructor_setting)
    yaml.add_multi_representer(MeterBase, multi_representer_obj)
    yaml.add_multi_representer(Model, multi_representer_obj)

    is_initialized = True

def multi_constructor_obj(loader, tag_suffix, node):
    """Callback used by PyYAML when a "!obj:" tag is encountered.
    See PyYAML documentation for details on the call signature.
    """
    yaml_src = yaml.serialize(node)
    mapping = loader.construct_mapping(node)

    if '.' not in tag_suffix:
        callable = eval(tag_suffix)
    else:
        callable = try_to_import(tag_suffix)

    proxy = Proxy(callable=callable, keywords=mapping, yaml_src=yaml_src)

    return proxy

def constructor_setting(loader, node):
    """Callback used by PyYAML when a "!setting" tag is encountered.
    See PyYAML documentation for details on the call signature.
    """
    yaml_src = yaml.serialize(node)
    value = loader.construct_scalar(node)
    proxy = Proxy(callable=Setting, keywords={"name": value}, yaml_src=yaml_src)

    return proxy

def multi_representer_obj(dumper, data):
    meter_type_name = data.__module__ + "." + data.__class__.__name__
    mapping = data.yaml_mapping()
    node = dumper.represent_mapping(u'!obj:{}'.format(meter_type_name), mapping)
    return node

def dump(meter):
    return yaml.dump(meter)
