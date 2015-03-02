import yaml
from collections import namedtuple

is_initialized = False

BaseProxy = namedtuple('BaseProxy', ['callable', 'keywords', 'yaml_src'])

class Proxy(BaseProxy):

    __slots__ = []

    def __hash__(self):
        """Return a hash based on the object ID (to avoid hashing unhashable
        namedtuple elements).
        """
        return hash(id(self))

def load(stream):
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
    return _instantiate(proxy_graph,bindings={})

def load_path(path):
    """Load a Meter specification written in YAML format. `path` is the path
    to the yaml file.
    """
    with open(path, 'r') as f:
        content = ''.join(f.readlines())
    return load(content)

def _instantiate_proxy_tuple(proxy,bindings):

    if proxy in bindings:
        return bindings[proxy]
    else:
        kwargs = {k: _instantiate(v, bindings)
                  for k, v in proxy.keywords.items()}
        obj = proxy.callable(**kwargs)

    bindings[proxy] = obj
    return bindings[proxy]

def _instantiate(proxy,bindings={}):

    if isinstance(proxy, Proxy):
        return _instantiate_proxy_tuple(proxy,bindings)
    elif isinstance(proxy, dict):
        return {k: _instantiate(v,bindings) for k, v in proxy.items()}
    elif isinstance(proxy, list):
        return [_instantiate(v,bindings) for v in proxy]
    else:
        return proxy

def try_to_import(tag_suffix):
    components = tag_suffix.split('.')
    modulename = '.'.join(components[:-1])

    exec('import %s' % modulename)
    obj = eval(tag_suffix)
    return obj

def initialize():
    """Add multi_constructors to yaml parser.
    """
    yaml.add_multi_constructor('!obj:', multi_constructor_obj)
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

    proxy = Proxy(callable=callable,keywords=mapping,yaml_src=yaml_src)

    return proxy
