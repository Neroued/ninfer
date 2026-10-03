"""Independent Draft 2020-12 conformance corpus for native semantic masks.

No native parser, normalizer, model, compilation or GPU work is performed here.
Expected compile errors encode declared implementation subsets, not observed
probe outcomes. Accepted-invalid results are always fatal in compare_schema_semantics.py.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


def scalar_tree(value):
    if isinstance(value,str):
        return not any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    if isinstance(value,list):
        return all(scalar_tree(item) for item in value)
    if isinstance(value,dict):
        return all(scalar_tree(key) and scalar_tree(item) for key,item in value.items())
    return True


def decode_wire(wire):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ValueError('Interoperable JSON subset rejects duplicate object keys')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON number: '+value)
    value = json.loads(wire,object_pairs_hook=pairs,parse_constant=constant)
    if not scalar_tree(value):
        raise ValueError('Interoperable JSON subset excludes unpaired surrogate strings')
    return value


def oracle(schema, wire, wire_bytes=None):
    try:
        if wire_bytes is not None:
            wire = bytes(wire_bytes).decode('utf-8','strict')
        value = decode_wire(wire)
    except (ValueError,UnicodeError) as error:
        return False, {'protocol_error':str(error)}
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        return False, {'schema_error':str(error), 'schema_valid':False}
    errors = list(Draft202012Validator(schema).iter_errors(value))
    return not errors, {'value':value, 'validation_errors':[
        {'path':list(error.absolute_path),'schema_path':list(error.absolute_schema_path),'message':error.message}
        for error in errors]}


def generate():
    cases = []
    def add(family,schema,value=None,*,wire=None,wire_bytes=None,compile_error=False,contract=None,
            serialization_subset=None,rejection_offset=None):
        if wire is None:
            wire = json.dumps(value,ensure_ascii=False,separators=(',',':'))
        accepted, details = oracle(schema,wire,wire_bytes)
        row = {'id':family+'-'+str(sum(item['family']==family for item in cases)), 'family':family,
               'schema':deepcopy(schema),'wire':wire, 'expected_accept':accepted,
               'expected_compile_error':compile_error,'oracle':details}
        if wire_bytes is not None:
            row['wire_bytes'] = wire_bytes
        if contract:
            row['compile_error_contract'] = contract
        if serialization_subset:
            row['declared_serialization_subset'] = serialization_subset
        if rejection_offset is not None:
            row['expected_rejection_offset_max'] = rejection_offset
        cases.append(row)
    string_array = {'type':'array','items':{'type':'string'},'uniqueItems':True}
    alphabet = ['a','b','é','e\u0301','😀','a,b']
    for length in range(4):
        for values in itertools.product(alphabet,repeat=length):
            add('unique-free-product',string_array,list(values))
    wires = [
        r'["a","\u0061"]',r'["\u0061","a"]',r'["é","\u00e9"]',r'["é","\u00E9"]',
        r'["😀","\ud83d\ude00"]',r'["\uD83D\uDE00","😀"]',r'["/","\/"]',
        r'["\n","\u000a"]',r'["\t","\u0009"]',r'["\"","\u0022"]',r'["\\","\u005c"]',
        r'["\\u0061","a"]',r'["a\u0000x","a\u0000y"]',r'["a\u0000x","a\u0000x"]',
        r'["[", "]", "{", "}"]',r'["<think>","</think>"]',r'["<think>","<think>"]',
        r'["é","e\u0301"]',r'["😀","😀\ufe0f"]',r'["","a"]',r'["",""]',
        r'["a", "b"]',r'["a", "a"]',r'["\uD800"]',r'["\uDC00"]',r'["\uD800x"]',
    ]
    for wire in wires:
        add('unique-wire-escaping',string_array,wire=wire)
    for value in [['a','a '],['a ','a'],[' ',''],['\n',''],['A','a']]:
        add('unique-codepoint-equality-no-trim',string_array,value)
    for raw in [b'["\xed\xa0\x80"]',b'["\xc0\xaf"]',b'["\xe0\x80\xaf"]',b'["\xf0\x80\x80\xaf"]',b'["\xf4\x90\x80\x80"]']:
        add('unique-invalid-utf8',string_array,wire='',wire_bytes=list(raw))
    for value in [None,False,0,{},[1],['a',1],[None,None]]:
        add('unique-string-type',string_array,value)
    # Multiple probe rows share the cached compiled schema. Each starts from
    # a fork, so state cannot leak across accepted/rejected trial sequences.
    for value in [['a'],['a'],['a','a'],['a'],['b'],['a','b']]:
        add('fork-independent-trials',string_array,value)
    for enum in [['a'],['a','ab'],['a','b'],['é','😀']]:
        schema = {'type':'array','items':{'type':'string','enum':enum},'uniqueItems':True}
        for values in [[],*[list(items) for length in range(1,len(enum)+2) for items in itertools.product(enum,repeat=length)],['foreign']]:
            add('unique-finite-enum',schema,values)
        exhausted = json.dumps(enum,ensure_ascii=False,separators=(',',':'))
        invalid = exhausted[:-1]+','+json.dumps(enum[0],ensure_ascii=False)+']'
        add('unique-finite-exhaustion-prefix',schema,wire=invalid,
            rejection_offset=len(exhausted[:-1].encode('utf-8')))
    finite = {'type':'array','items':{'type':'string','enum':['a','ab']},'uniqueItems':True,'minItems':2,'maxItems':2}
    for value in [[],['a'],['a','ab'],['ab','a'],['a','a'],['ab','ab'],['a','ab','a']]:
        add('unique-finite-count',finite,value)
    for wire in [r'["a","\u0061"]',r'["a","\u0061b"]',r'["é","\u00E9"]',r'["😀","\uD83D\uDE00"]']:
        enum = ['a','ab'] if 'a' in wire else ['é','😀']
        schema = {'type':'array','items':{'type':'string','enum':enum},'uniqueItems':True}
        add('unique-finite-escaped',schema,wire=wire,
            serialization_subset='Vendor enum CFG may emit only canonical literal spelling; semantic layer must still compare decoded strings correctly.')
    for maximum in [0,1]:
        for items in [True,{'type':'number'},{'type':'object'},{'type':['string','null']},{'type':'string','pattern':'^[ab]$'}]:
            schema = {'type':'array','items':items,'uniqueItems':True,'maxItems':maximum}
            for value in [[],['a'],[1],[{}],[None],['a','a'],[1,1]]:
                add('unique-vacuous-max-items',schema,value)
    disabled = {'type':'array','uniqueItems':False,'items':True}
    for value in [[],['a','a'],[1,1],[True,1],[{},{}],[None,None]]:
        add('unique-disabled',disabled,value)
    nullable = {'type':['array','null'],'items':{'type':'string'},'uniqueItems':True}
    for value in [None,[],['a'],['a','a'],['a','b'],1,{}]:
        add('unique-root-nullable',nullable,value)
    nullable_any = {'anyOf':[deepcopy(string_array),{'type':'null'}]}
    for value in [None,[],['a'],['a','a']]:
        add('unique-root-nullable-anyof',nullable_any,value)
    for maximum in [0,1,2]:
        schema = {'type':'object','maxProperties':maximum,'additionalProperties':True}
        for value in [{},{'a':1},{'a':1,'b':2},{'a':1,'b':2,'c':3}]:
            add('max-properties-open',schema,value)
        schema = {'type':'object','properties':{'a':{'type':'string'},'b':{'type':'string'},'c':{'type':'string'}},
                  'maxProperties':maximum,'additionalProperties':False}
        for value in [{},{'a':'x'},{'b':'x'},{'a':'x','b':'y'},{'a':'x','b':'y','c':'z'},{'extra':'x'}]:
            add('max-properties-declared',schema,value)
        schema = {'type':['object','null'],'maxProperties':maximum,'additionalProperties':True}
        for value in [None,{}, {'a':1},{'a':1,'b':2},['a']]:
            add('max-properties-nullable',schema,value)
    independent = {'type':'object','properties':{'left':deepcopy(string_array),'right':deepcopy(string_array)},
                   'required':['left','right'],'additionalProperties':False}
    for value in [{'left':['a'],'right':['a']},{'left':['a','b'],'right':['a','b']},
                  {'left':['a','a'],'right':['a']},{'left':['a'],'right':['a','a']}]:
        add('unique-independent-fields',independent,value)
    nested = {'type':'array','items':deepcopy(string_array)}
    for value in [[],[[]],[['a'],['a']],[['a','b'],['a','b']],[['a','a']],[['a'],['a','a']]]:
        add('unique-nested-arrays',nested,value)
    records = {'type':'array','items':{'type':'object','properties':{'labels':deepcopy(string_array)},
                                      'required':['labels'],'additionalProperties':False}}
    for value in [[{'labels':['a']},{'labels':['a']}],[{'labels':['a','a']}],[{'labels':['a']},{'labels':['a','a']}]]:
        add('unique-record-array-field',records,value)
    additional = {'type':'object','additionalProperties':deepcopy(string_array)}
    for value in [{},{'x':['a'],'y':['a']},{'x':['a','b'],'y':['b','a']},{'x':['a','a']}, {'x':['a'],'y':['a','a']}]:
        add('unique-additional-properties',additional,value)
    escaped_key = {'type':'object','properties':{'a/b~c':deepcopy(string_array)},'required':['a/b~c'],'additionalProperties':False}
    for value in [{'a/b~c':['a','b']},{'a/b~c':['a','a']}]:
        add('unique-json-pointer-property',escaped_key,value)
    reference = {'$defs':{'strings':deepcopy(string_array)},'type':'object','properties':{
        'left':{'$ref':'#/$defs/strings'},'right':{'$ref':'#/$defs/strings'}},'required':['left','right'],'additionalProperties':False}
    for value in [{'left':['a'],'right':['a']},{'left':['a','a'],'right':[]},{'left':[],'right':['a','a']}]:
        add('unique-local-reference',reference,value)
    itemref = {'$defs':{'text':{'type':'string'}},'type':'array','items':{'$ref':'#/$defs/text'},'uniqueItems':True}
    for value in [[],['a'],['a','a'],['a','b'],[1]]:
        add('unique-item-local-reference',itemref,value)
    obj = {'type':'object','required':['parent'],'anyOf':[
        {'type':'object','properties':{'parent':{'type':'string'},'left':{'type':'integer'}},
         'required':['left'],'additionalProperties':False},
        {'type':'object','properties':{'parent':{'type':'string'},'right':{'type':'string'}},
         'required':['right'],'additionalProperties':False}]}
    for value in [None,1,[],{}, {'parent':'p'},{'parent':'p','left':1},{'left':1},
                  {'parent':'p','right':'r'},{'right':'r'},{'parent':'p','left':'bad'},{'parent':'p','left':1,'extra':True}]:
        add('anyof-common-required-type',obj,value)
    prune = {'type':'object','required':['parent'],'anyOf':[
        {'type':'string'}, {'type':'object','properties':{'parent':{'type':'string'}},'additionalProperties':False}]}
    for value in ['a',{}, {'parent':'p'}, {'parent':1}]:
        add('anyof-type-pruning',prune,value)
    for schema in [
        {'type':'object','anyOf':[{}]},
        {'type':['integer','null'],'anyOf':[{'type':'number'},{'type':'null'}]},
        {'type':'string','anyOf':[{'type':'string'},{'type':'integer'}]},
        {'type':'object','required':['parent'],'anyOf':[{'type':'object','properties':{'parent':{'type':'string'}},'additionalProperties':False},True]},
    ]:
        for value in [None,False,0,1,'a',[],{}, {'parent':'p'},{'parent':1}]:
            add('anyof-type-intersection',schema,value)
    required_scalar = {'required':['parent'],'anyOf':[{'type':'string'},
        {'type':'object','properties':{'parent':{'type':'string'}},'additionalProperties':False}]}
    for value in ['a',{}, {'parent':'p'},{'parent':1},None]:
        add('anyof-required-does-not-constrain-scalars',required_scalar,value)
    refany = {'$defs':{'record':{'type':'object','properties':{'parent':{'type':'string'},'value':{'type':'integer'}},
                                  'required':['value'],'additionalProperties':False}},
              'type':'object','required':['parent'],'anyOf':[{'$ref':'#/$defs/record'}]}
    for value in [{'parent':'p','value':1},{'value':1},{'parent':'p','value':'x'}]:
        add('anyof-local-ref-distribution',refany,value)
    unique_branch = {'type':'object','properties':{'kind':{'type':'string','const':'unique'},'tags':deepcopy(string_array)},
                    'required':['kind','tags'],'additionalProperties':False}
    plain_branch = {'type':'object','properties':{'kind':{'type':'string','const':'plain'},'tags':{'type':'array','items':{'type':'string'}}},
                   'required':['kind','tags'],'additionalProperties':False}
    discriminated = {'anyOf':[unique_branch,plain_branch]}
    for value in [{'kind':'unique','tags':['a','b']},{'kind':'unique','tags':['a','a']},
                  {'kind':'plain','tags':['a','a']},{'kind':'other','tags':['a']}]:
        add('unique-early-discriminator',discriminated,value)
    add('unique-late-discriminator',discriminated,wire='{"tags":["a","a"],"kind":"plain"}',
        serialization_subset='Declared-property-order decoding subset requires early discriminator before the unique array.')
    common = {'anyOf':[deepcopy(unique_branch),{**deepcopy(unique_branch),'properties':{
        'kind':{'type':'string','const':'other'},'tags':deepcopy(string_array)}}]}
    for value in [{'kind':'unique','tags':['a','b']},{'kind':'unique','tags':['a','a']},{'kind':'other','tags':['a','b']}]:
        add('unique-common-anyof-domain',common,value)
    unsupported = [
        ('unique-nonstring',{'type':'array','items':{'type':'number'},'uniqueItems':True}, [[],[1],[1,1]],
         'Nonvacuous uniqueness currently supports explicit string items only.'),
        ('unique-nullable-item',{'type':'array','items':{'type':['string','null']},'uniqueItems':True}, [[],['a'],[None,None]],
         'Item type must be proven pure string for nonvacuous semantic uniqueness.'),
        ('unique-string-intersection',{'type':'array','items':{'type':'string','minLength':1},'uniqueItems':True}, [[],['a'],['a','a']],
         'Nonvacuous item length/pattern/format intersections are outside the current semantic mask subset.'),
        ('unique-prefix-tuple',{'type':'array','prefixItems':[{'type':'string'}],'items':{'type':'string'},'uniqueItems':True}, [[],['a'],['a','a']],
         'Nonvacuous tuple prefix uniqueness is outside the current semantic mask subset.'),
        ('unique-mixed-open-anyof',{'anyOf':[deepcopy(string_array),{'type':'array','items':{'type':'string'}}]}, [[],['a','a'],['a','b']],
         'Mixed unique/nonunique open array alternatives have no early provable discriminator.'),
        ('unique-different-finite-anyof',{'anyOf':[
            {'type':'array','items':{'type':'string','enum':['a','b']},'uniqueItems':True},
            {'type':'array','items':{'type':'string','enum':['a','c']},'uniqueItems':True}]}, [[],['a','b'],['a','c'],['a','a']],
         'Different finite domains without early discriminator are conservatively unsupported.'),
        ('anyof-unimplemented-siblings',{'type':'array','uniqueItems':True,'items':{'type':'string'},'anyOf':[{}]}, [[],['a'],['a','a']],
         'AnyOf sibling normalization currently distributes common type/required only, not arbitrary assertions.'),
        ('anyof-const-required',{'type':'object','required':['parent'],'anyOf':[{'const':{'parent':'p'}}]}, [{'parent':'p'},{}],
         'Required distribution through const/enum lacks a proven literal intersection.'),
    ]
    for family,schema,values,contract in unsupported:
        for value in values:
            add(family,schema,value,compile_error=True,contract=contract)
    unsatisfiable = {'type':'object','required':['missing'],'anyOf':[{'type':'object','properties':{'present':{'type':'string'}},'additionalProperties':False}]}
    for value in [{},{'missing':'x'},{'present':'x'},{'present':'x','missing':'x'}]:
        add('anyof-unsatisfiable-closed-parent',unsatisfiable,value,compile_error=True,
            contract='Provably unsatisfiable root is explicitly rejected, not emitted as a weakened branch.')
    type_conflict = {'type':'string','anyOf':[{'type':'object'}]}
    for value in ['x',{},1]:
        add('anyof-unsatisfiable-type',type_conflict,value,compile_error=True,
            contract='Empty type intersection makes the root language unsatisfiable.')
    no_ref_id = {'type':'object','properties':{'tags':{'$id':'https://synthetic.invalid/local.json',**deepcopy(string_array)}},
                 'required':['tags'],'additionalProperties':False}
    for value in [{'tags':['a','b']},{'tags':['a','a']}]:
        add('nested-id-no-reference',no_ref_id,value)
    scoped_ref = {'$defs':{'resource':{'$id':'https://synthetic.invalid/child.json',
        '$defs':{'text':{'type':'string'}},'type':'object','properties':{'value':{'$ref':'#/$defs/text'}},
        'required':['value'],'additionalProperties':False}},
        'type':'object','properties':{'child':{'$ref':'#/$defs/resource'}},'required':['child'],'additionalProperties':False}
    for value in [{'child':{'value':'a'}},{'child':{'value':1}}]:
        add('nested-id-reference-scope',scoped_ref,value,compile_error=True,
            contract='Resource identifiers containing references need URI-scoped resolution; fail closed until implemented.')
    metadata = {**deepcopy(string_array),'uniqueItems':False,
        'description':'uniqueItems in annotations is data, not a nested assertion',
        'default':['a','a'],'examples':[{'type':'array','uniqueItems':True,'items':{'type':'number'}}],
        '$comment':'{"anyOf":[],"$id":"not-a-resource"}'}
    for value in [['a','a'],['a','b'],[1]]:
        add('metadata-paths-opaque',metadata,value)
    names = {'type':'object','properties':{'uniqueItems':{'type':'boolean'},
        'items':{'type':'array','items':{'type':'string'},'uniqueItems':False},
        'description':deepcopy(string_array)},'required':['uniqueItems','items','description'],'additionalProperties':False}
    for value in [{'uniqueItems':True,'items':['a','a'],'description':['a','b']},
                  {'uniqueItems':False,'items':['a','b'],'description':['a','a']}]:
        add('metadata-looking-business-fields',names,value)
    literal = {'uniqueItems':True,'items':['a','a'],'$id':'literal-data','$ref':'literal-data'}
    opaque = {'type':'object','const':literal}
    add('metadata-keywords-in-const-data',opaque,literal)
    for wire in ['<think>["a","a"]</think>["a","b"]','["a","b"]<tool>opaque</tool>',
                 '<tool>{"description":["a","a"]}</tool>["a"]']:
        add('direct-probe-envelope-is-not-json',string_array,wire=wire)
    # Independent new families: preserve IDs of the original 646 cases.
    for domain in [['a','ab'],['é','😀'],[''],['','a']]:
        schema = {'type':'array','items':{'enum':domain},'uniqueItems':True}
        for value in [[],[domain[0]],domain,[domain[0],domain[0]],['foreign'],[None]]:
            add('unique-untyped-string-enum',schema,value)
        exhausted = json.dumps(domain,ensure_ascii=False,separators=(',',':'))
        add('unique-untyped-enum-exhaustion',schema,
            wire=exhausted[:-1]+','+json.dumps(domain[0],ensure_ascii=False)+']',
            rejection_offset=len(exhausted[:-1].encode('utf-8')))
        unsat = {**deepcopy(schema),'minItems':len(set(domain))+1}
        for value in [[],domain,[domain[0]]*(len(set(domain))+1)]:
            add('unique-untyped-enum-unsatisfiable-cardinality',unsat,value,compile_error=True,
                contract='minItems exceeds the proven finite distinct string domain; compile must reject before a dead prefix.')
    for literal in ['a','é','😀','']:
        schema = {'type':'array','items':{'const':literal},'uniqueItems':True}
        for value in [[],[literal],[literal,literal],['foreign']]:
            add('unique-untyped-const-string',schema,value)
        for value in [[],[literal],[literal,literal]]:
            add('unique-untyped-const-unsatisfiable-min2',{**deepcopy(schema),'minItems':2},value,
                compile_error=True,contract='One proven constant string cannot satisfy two unique items.')
    for items in [{'enum':['a',1]},{'enum':[None,'a']},{'const':1},{}]:
        schema = {'type':'array','items':items,'uniqueItems':True}
        for value in [[],['a'],['a','a'],[1,1]]:
            add('unique-untyped-without-pure-string-proof',schema,value,compile_error=True,
                contract='No pure-string finite domain proof for a nonvacuous unique array; fail closed.')
    empty_enum = {'type':'array','items':{'enum':[]},'uniqueItems':True}
    for value in [[],['a']]:
        add('unique-untyped-empty-enum-invalid-schema',empty_enum,value,compile_error=True,
            contract='Empty enum violates the Draft 2020-12 metaschema and cannot be treated as a normal finite proof.')
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    cases = generate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(item,ensure_ascii=True)+'\n' for item in cases),encoding='utf-8')
    stats = {'cases':len(cases),'expected_accept':sum(item['expected_accept'] for item in cases),
        'expected_reject':sum(not item['expected_accept'] for item in cases),
        'expected_compile_error':sum(item['expected_compile_error'] for item in cases),
        'declared_serialization_subset':sum('declared_serialization_subset' in item for item in cases),
        'oracle':'Unmodified original schema with independent jsonschema Draft202012Validator; strict interoperable JSON decoding',
        'reasoning_envelope_contract':'Raw envelope wires must reject in direct JSON compiler probe. Wrapped reasoning/tool/fork rollback contracts belong to the native C++ adapter tests.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
