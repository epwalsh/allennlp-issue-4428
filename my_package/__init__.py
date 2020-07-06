from allennlp.data import Instance, DatasetReader
from allennlp.data.fields import LabelField


@DatasetReader.register("my_reader")
class MyReader(DatasetReader):
    def _read(self, path):
        for i in range(10):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int) -> Instance:
        return Instance({"index": LabelField(index, skip_indexing=True)})
