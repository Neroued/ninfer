"""NInfer v3 representation and file tools.

formats defines stored numbers; layouts defines their byte geometry; codecs
packs and decodes exact words. tensor_output maps row blocks to complete objects.
schema describes the directory; framing, reader, and writer own the file set.

The file and directory modules use the standard library. Tensor dependencies
are confined to explicitly imported codecs and tensor_output.
"""
