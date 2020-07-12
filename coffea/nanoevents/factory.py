from weakref import WeakValueDictionary
import numpy
import awkward1
import uproot
from coffea.nanoevents.indirection import distinctParent, children


def _with_length(array: awkward1.layout.VirtualArray, length: int):
    return awkward1.layout.VirtualArray(
        array.generator.with_length(length),
        array.cache,
        array.cache_key,
        array.identities,
        array.parameters,
    )


class NanoEventsFactory:
    default_mixins = {
        "CaloMET": "MissingET",
        "ChsMET": "MissingET",
        "GenMET": "MissingET",
        "MET": "MissingET",
        "METFixEE2017": "MissingET",
        "PuppiMET": "MissingET",
        "RawMET": "MissingET",
        "TkMET": "MissingET",
        # pseudo-lorentz: pt, eta, phi, mass=0
        "IsoTrack": "PtEtaPhiMCollection",
        "SoftActivityJet": "PtEtaPhiMCollection",
        "TrigObj": "PtEtaPhiMCollection",
        # True lorentz: pt, eta, phi, mass
        "FatJet": "FatJet",
        "GenDressedLepton": "PtEtaPhiMCollection",
        "GenJet": "PtEtaPhiMCollection",
        "GenJetAK8": "FatJet",
        "Jet": "Jet",
        "LHEPart": "PtEtaPhiMCollection",
        "SV": "PtEtaPhiMCollection",
        "SubGenJetAK8": "PtEtaPhiMCollection",
        "SubJet": "PtEtaPhiMCollection",
        # Candidate: lorentz + charge
        "Electron": "Electron",
        "Muon": "Muon",
        "Photon": "Photon",
        "Tau": "Tau",
        "GenVisTau": "GenVisTau",
        # special
        "GenPart": "GenParticle",
    }
    """Default configuration for mixin types, based on the collection name."""
    nested_items = {
        "FatJet_subJetIdxG": ["FatJet_subJetIdx1G", "FatJet_subJetIdx2G"],
        "Jet_muonIdxG": ["Jet_muonIdx1G", "Jet_muonIdx2G"],
        "Jet_electronIdxG": ["Jet_electronIdx1G", "Jet_electronIdx2G"],
    }
    """Default nested collections, where nesting is accomplished by a fixed-length set of indexers"""
    special_items = {
        "GenPart_distinctParentIdxG": (
            distinctParent,
            ("GenPart_genPartIdxMotherG", "GenPart_pdgId"),
        ),
        "GenPart_childrenIdxG": (children, ("oGenPart", "GenPart_genPartIdxMotherG")),
        "GenPart_distinctChildrenIdxG": (
            children,
            ("oGenPart", "GenPart_distinctParentIdxG"),
        ),
    }
    """Default special arrays, where the callable and input arrays are specified in the value"""
    _active = WeakValueDictionary()

    def __init__(
        self,
        file,
        treename="Events",
        entrystart=None,
        entrystop=None,
        cache=None,
        mixin_map=None,
        metadata=None,
    ):
        if not isinstance(file, uproot.rootio.ROOTDirectory):
            file = uproot.open(file)
        self._tree = file[treename]
        self._entrystart, self._entrystop = uproot.tree._normalize_entrystartstop(
            self._tree.numentries, entrystart, entrystop
        )
        self._keyprefix = "/".join(
            [
                file._context.uuid.hex(),
                treename,
                str(self._entrystart),
                str(self._entrystop),
            ]
        )
        NanoEventsFactory._active[self._keyprefix] = self

        if cache is None:
            cache = awkward1.layout.ArrayCache({})
        else:
            cache = awkward1.layout.ArrayCache(cache)
        self._cache = cache

        self._mixin_map = {}
        self._mixin_map.update(self.default_mixins)
        if mixin_map is not None:
            self._mixin_map.update(mixin_map)

        self._metadata = metadata  # TODO: JSON only?
        self._branches_read = set()
        self._events = None

    @classmethod
    def _instance(cls, key):
        try:
            return cls._active[key]
        except KeyError:
            raise RuntimeError(
                "NanoEventsFactory instance was lost, cross-references are now invalid"
            )

    @classmethod
    def get_events(cls, key):
        return cls._instance(key).events()

    @classmethod
    def get_cache(cls, key):
        return cls._instance(key)._cache

    def __len__(self):
        return self._entrystop - self._entrystart

    def reader(self, branch_name, parameters):
        self._branches_read.add(branch_name)
        return awkward1.layout.NumpyArray(
            self._tree[branch_name].array(
                entrystart=self._entrystart, entrystop=self._entrystop, flatten=True
            ),
            parameters=parameters,
        )

    def _array(self, branch_name):
        interpretation = uproot.interpret(self._tree[branch_name])
        if isinstance(interpretation, uproot.asjagged):
            dtype = interpretation.content.type
            length = None
        else:
            dtype = interpretation.type
            length = len(self)
        parameters = {"__doc__": self._tree[branch_name].title.decode("ascii")}
        # use hint to resolve platform-dependent format
        formhint = awkward1.forms.Form.fromjson('"%s"' % dtype)
        form = awkward1.forms.NumpyForm(
            [], formhint.itemsize, formhint.format, parameters=parameters
        )
        generator = awkward1.layout.ArrayGenerator(
            self.reader, (branch_name, parameters), {}, form=form, length=length,
        )
        source = "file"
        return awkward1.layout.VirtualArray(
            generator,
            self._cache,
            cache_key="/".join([self._keyprefix, source, branch_name]),
            parameters=parameters,
        )

    def _getoffsets(self, counts_name, parameters):
        counts = self.reader(counts_name, parameters)
        offsets = numpy.empty(len(counts) + 1, numpy.uint32)
        offsets[0] = 0
        numpy.cumsum(counts, out=offsets[1:])
        return awkward1.layout.NumpyArray(offsets, parameters=parameters,)

    def _counts2offsets(self, virtual_counts):
        generator = virtual_counts.generator
        generator = generator.with_callable(self._getoffsets).with_length(
            generator.length + 1
        )
        return awkward1.layout.VirtualArray(
            generator,
            virtual_counts.cache,
            virtual_counts.cache_key + "/counts2offsets",
            virtual_counts.identities,
            virtual_counts.parameters,
        )

    def _local2global(self, index, source_offsets, target_offsets):
        def globalindex():
            gidx = awkward1.Array(
                awkward1.layout.ListOffsetArray32(
                    awkward1.layout.Index32(source_offsets), index.generator(),
                )
            )
            gidx = gidx.mask[gidx >= 0] + target_offsets[:-1]
            return awkward1.fill_none(awkward1.flatten(gidx), -1)

        generator = awkward1.layout.ArrayGenerator(
            globalindex,
            (),
            {},
            form=awkward1.forms.Form.fromjson('"int64"'),
            length=index.generator.length,
        )
        return awkward1.layout.VirtualArray(
            generator,
            index.cache,
            index.cache_key + "/local2global",
            index.identities,
            index.parameters,
        )

    def _nestedindex(self, indexers):
        def nestedindex():
            # idx = awkward1.concatenate([idx[:, None] for idx in indexers], axis=1)
            n = len(indexers)
            out = numpy.empty(n * len(indexers[0]), dtype="int64")
            for i, idx in enumerate(indexers):
                out[i::n] = idx
            offsets = numpy.arange(0, len(out) + 1, n, dtype=numpy.int64)
            return awkward1.layout.ListOffsetArray64(
                awkward1.layout.Index64(offsets), awkward1.layout.NumpyArray(out),
            )

        form = awkward1.forms.Form.fromjson(
            '{"class": "ListOffsetArray64", "offsets": "i64", "content": "int64"}'
        )
        generator = awkward1.layout.ArrayGenerator(
            nestedindex, (), {}, form=form, length=indexers[0].generator.length,
        )
        return awkward1.layout.VirtualArray(
            generator, self._cache, indexers[0].cache_key + "/nestedindex",
        )

    def _listarray(self, offsets, content, recordparams):
        offsets = awkward1.layout.Index32(offsets)
        length = offsets[-1]
        return awkward1.layout.ListOffsetArray32(
            offsets,
            awkward1.layout.RecordArray(
                {k: _with_length(v, length) for k, v in content.items()},
                parameters=recordparams,
            ),
        )

    def events(self):
        if self._events is not None:
            return self._events

        arrays = {branch_name.decode("ascii") for branch_name in self._tree.keys()}

        # parse into high-level records (collections, list collections, and singletons)
        collections = set(k.split("_")[0] for k in arrays)
        collections -= set(
            k for k in collections if k.startswith("n") and k[1:] in collections
        )

        arrays = {k: self._array(k) for k in arrays}

        # Create offsets virtual arrays
        for name in collections:
            if "n" + name in arrays:
                arrays["o" + name] = self._counts2offsets(arrays["n" + name])

        # Create global index virtual arrays for indirection
        for name in collections:
            indexers = filter(lambda k: k.startswith(name) and "Idx" in k, arrays)
            for k in list(indexers):
                target = k[len(name) + 1 : k.find("Idx")]
                target = target[0].upper() + target[1:]
                if target not in collections:
                    raise RuntimeError(
                        "Parsing indexer %s, expected to find collection %s but did not"
                        % (k, target)
                    )
                arrays[k + "G"] = self._local2global(
                    arrays[k], arrays["o" + name], arrays["o" + target]
                )

        # Create nested indexer from Idx1, Idx2, ... arrays
        for name, indexers in self.nested_items.items():
            if all(idx in arrays for idx in indexers):
                arrays[name] = self._nestedindex([arrays[idx] for idx in indexers])

        # Create any special arrays
        for name, (fcn, args) in self.special_items.items():
            if all(k in arrays for k in args):
                generator = fcn(*(arrays[k] for k in args))
                arrays[name] = awkward1.layout.VirtualArray(
                    generator,
                    self._cache,
                    cache_key="/".join([self._keyprefix, fcn.__name__, name]),
                )

        def collectionfactory(name):
            mixin = self._mixin_map.get(name, "NanoCollecton")
            if "o" + name in arrays:
                # list collection
                offsets = arrays["o" + name]
                content = {
                    k[len(name) + 1 :]: arrays[k]
                    for k in arrays
                    if k.startswith(name + "_")
                }
                recordparams = {
                    "__doc__": offsets.parameters["__doc__"],
                    "__record__": mixin,
                    "events_key": self._keyprefix,
                    "collection_name": name,
                }
                form = awkward1.forms.ListOffsetForm(
                    "i32",
                    awkward1.forms.RecordForm(
                        {k: v.form for k, v in content.items()}, parameters=recordparams
                    ),
                )
                generator = awkward1.layout.ArrayGenerator(
                    self._listarray,
                    (offsets, content, recordparams),
                    {},
                    form=form,
                    length=len(self),
                )
                source = "runtime"
                return awkward1.layout.VirtualArray(
                    generator,
                    self._cache,
                    cache_key="/".join([self._keyprefix, source, name]),
                    parameters=recordparams,
                )
            elif name in arrays:
                # singleton
                return arrays[name]
            else:
                # simple collection
                content = {
                    k[len(name) + 1 :]: arrays[k]
                    for k in arrays
                    if k.startswith(name + "_")
                }
                return awkward1.layout.RecordArray(
                    content,
                    parameters={
                        "__record__": mixin,
                        "events_key": self._keyprefix,
                        "collection_name": name,
                    },
                )

        events = awkward1.layout.RecordArray(
            {name: collectionfactory(name) for name in collections},
            parameters={
                "__record__": "NanoEvents",
                "__doc__": self._tree.title.decode("ascii"),
                "events_key": self._keyprefix,
                "metadata": self._metadata,
            },
        )

        self._events = awkward1.Array(events)
        return self._events
