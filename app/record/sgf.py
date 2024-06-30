#!/usr/bin/python
# -*- coding: utf-8 -*-

from io import StringIO

# map from numerical coordinates to letters used by SGF
SGF_POS = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class Collection:
    def __init__(self, parser=None):
        self._parser = parser
        if parser:
            self.setup()
        self._children = []

    def setup(self):
        self._parser.start_gametree = self._start_gametree

    def _start_gametree(self):
        self._children.append(GameTree(self, self._parser))

    def __len__(self):
        return len(self._children)

    def __getitem__(self, k):
        return self._children[k]

    def __iter__(self):
        return iter(self._children)

    @property
    def children(self):
        return self._children

    def output(self, f):
        for child in self._children:
            child.output(f)


class GameTree(object):
    def __init__(self, parent, parser=None):
        self._parent = parent
        self._parser = parser
        if parser:
            self.setup()
        self._nodes = []
        self._children = []

    def setup(self):
        self._parser.start_gametree = self._start_gametree
        self._parser.end_gametree = self._end_gametree
        self._parser.start_node = self._start_node

    def _start_node(self):
        if len(self._nodes) > 0:
            previous = self._nodes[-1]
        elif self._parent.__class__ == GameTree:
            previous = self._parent.nodes[-1]
        else:
            previous = None
        node = Node(self, previous, self._parser)
        if len(self._nodes) == 0:
            node.first = True
            if self._parent.__class__ == GameTree:
                if len(previous.variations) > 0:
                    previous.variations[-1].next_variation = node
                    node.previous_variation = previous.variations[-1]
                previous.variations.append(node)
            else:
                if len(self._parent.children) > 1:
                    node.previous_variation = self._parent.children[-2].nodes[0]
                    self._parent.children[-2].nodes[0].next_variation = node

        self._nodes.append(node)

    def _start_gametree(self):
        self._children.append(GameTree(self, self._parser))

    def _end_gametree(self):
        self._parent.setup()

    def __iter__(self):
        return NodeIterator(self._nodes[0])

    @property
    def nodes(self):
        return self._nodes

    @property
    def root(self):
        # @@@ technically for this to be root, self.parent must be a Collection
        return self._nodes[0]

    @property
    def rest(self):
        class _:
            def __iter__(_):
                return NodeIterator(self._nodes[0].next)
        if self._nodes[0].next:
            return _()
        else:
            return None

    def output(self, f):
        f.write('(')
        for node in self._nodes:
            node.output(f)
        for child in self._children:
            child.output(f)
        f.write(')')


class Node:
    def __init__(self, parent, previous, parser=None):
        self._parent = parent
        self._previous = previous
        self._parser = parser
        if parser:
            self.setup()
        self._properties = {}
        self._next = None
        self._previous_variation = None
        self._next_variation = None
        self._first = False
        self._variations = []
        if previous and not previous._next:
            previous._next = self

    def setup(self):
        self._parser.start_property = self._start_property
        self._parser.add_prop_value = self._add_prop_value
        self._parser.end_property = self._end_property
        self._parser.end_node = self._end_node

    def _start_property(self, identifier):
        # @@@ check for duplicates
        self._current_property = identifier
        self._current_prop_value = []

    def _add_prop_value(self, value):
        self._current_prop_value.append(value)

    def _end_property(self):
        self._properties[self._current_property] = self._current_prop_value

    def _end_node(self):
        self._parent.setup()

    @property
    def first(self):
        return self._first

    @first.setter
    def first(self, first):
        self._first = first

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, n):
        self._next = n

    @property
    def previous_variation(self):
        return self._previous_variation

    @previous_variation.setter
    def previous_variation(self, previous_variation):
        self._previous_variation = previous_variation

    @property
    def next_variation(self):
        return self._next_variation

    @next_variation.setter
    def next_variation(self, next_variation):
        self._next_variation = next_variation

    @property
    def variations(self):
        return self._variations

    @property
    def properties(self):
        return self._properties

    def output(self, f):
        f.write(';')
        for key, values in sorted(self._properties.items()):
            f.write(key)
            for value in values:
                if '\\' in value:
                    value = '\\\\'.join(value.split('\\'))
                if ']' in value:
                    value = '\\]'.join(value.split(']'))
                f.write('[%s]' % value)


class NodeIterator:
    def __init__(self, start_node):
        self._node = start_node

    def __next__(self):
        if self._node:
            node = self._node
            self._node = node.next
            return node
        else:
            raise StopIteration()

    next = __next__  # Python 2


class ParseException(Exception):
    pass


class Parser:
    def __init__(self):
        self.start_gametree = None
        self.start_node = None
        self.start_property = None
        self.add_prop_value = None
        self.end_property = None
        self.end_node = None
        self.end_gametree = None

    def parse(self, sgf_string):

        def whitespace(ch):
            return ch in ' \t\r\n'

        def ucletter(ch):
            return ch in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        state = 0
        prop_ident = ''
        prop_value = ''
        ch = ''

        square_brackets = 0
        for ch in sgf_string:
            if state == 0:
                if whitespace(ch):
                    state = 0
                elif ch == '(':
                    self.start_gametree()
                    state = 1
                else:
                    state = 0  # ignore everything up to first (
                    # raise ParseException(ch, state)
            elif state == 1:
                if whitespace(ch):
                    state = 1
                elif ch == ';':
                    self.start_node()
                    state = 2
                else:
                    raise ParseException(ch, state)
            elif state == 2:
                if whitespace(ch):
                    state = 2
                elif ucletter(ch):
                    prop_ident = ch
                    state = 3
                elif ch == ';':
                    self.end_node()
                    self.start_node()
                    state = 2
                elif ch == '(':
                    self.end_node()
                    self.start_gametree()
                    state = 1
                elif ch == ')':
                    self.end_node()
                    self.end_gametree()
                    state = 4
                else:
                    raise ParseException(ch, state)
            elif state == 3:
                if ucletter(ch):
                    prop_ident = prop_ident + ch
                    state = 3
                elif ch == '[':
                    self.start_property(prop_ident)
                    prop_value = ''
                    state = 5
                else:
                    raise ParseException(ch, state)
            elif state == 4:
                if ch == ')':
                    self.end_gametree()
                    state = 4
                elif whitespace(ch):
                    state = 4
                elif ch == '(':
                    self.start_gametree()
                    state = 1
                else:
                    raise ParseException(ch, state)
            elif state == 5:
                if ch == '\\':
                    state = 6
                elif ch == ']' and square_brackets == 0:
                    self.add_prop_value(prop_value)
                    state = 7
                else:
                    prop_value = prop_value + ch
                    if ch == '[':
                        square_brackets += 1
                    elif ch == ']':
                        square_brackets -= 1
            elif state == 6:
                prop_value = prop_value + ch
                state = 5
            elif state == 7:
                if whitespace(ch):
                    state = 7
                elif ch == '[':
                    prop_value = ''
                    state = 5
                elif ch == ';':
                    self.end_property()
                    self.end_node()
                    self.start_node()
                    state = 2
                elif ucletter(ch):
                    self.end_property()
                    prop_ident = ch
                    state = 3
                elif ch == ')':
                    self.end_property()
                    self.end_node()
                    self.end_gametree()
                    state = 4
                elif ch == '(':
                    self.end_property()
                    self.end_node()
                    self.start_gametree()
                    state = 1
                else:
                    raise ParseException(ch, state)
            else:
                # only a programming error could get here
                raise Exception(state)  # pragma: no cover

        if state != 4:
            raise ParseException(ch, state)


def parse(sgf_string):
    parser = Parser()
    collection = Collection(parser)
    parser.parse(sgf_string)
    return collection


def main():
    example = '(;FF[4]GM[1]SZ[19];B[aa];W[bb];B[cc];W[dd];B[ad];W[bd])'
    collection = parse(example)

    for game in collection:
        for _ in game:
            pass

    with StringIO() as out:
        collection[0].nodes[1].output(out)
        assert out.getvalue() == ';B[aa]'

    with StringIO() as out:
        collection.output(out)
        assert out.getvalue() == example

    example2 = '(;FF[4]GM[1]SZ[19];B[aa];W[bb](;B[cc];W[dd];B[ad];W[bd])' \
               '(;B[hh];W[hg]))'
    collection = parse(example2)

    with StringIO() as out:
        collection.output(out)
        assert out.getvalue() == example2

    count = 0
    for _ in collection[0].rest:
        count += 1
    assert count == 6

    example3 = '(;C[foo\\]\\\\])'
    collection = parse(example3)

    with StringIO() as out:
        collection.output(out)
        assert out.getvalue() == example3

    parse('foo(;)')  # junk before first ( is supported

    parse('(  ;)')  # whitespace after ( is allowed

    parse('(;;)')  # a node after an empty node is allowed
    parse('(;(;))')  # a gametree after an empty node is allowed

    parse('(;PW[[梵音]选手01]PB[[伯园]选手01]DT[2008-6-23]RE[白胜]SW[True]SZ[15];B[hh];W[hi];)')

    # errors

    try:
        parse('()')  # games must have a node
        assert False  # pragma: no cover
    except ParseException:
        pass

    try:
        parse('(W[tt])')  # a property has to be in a node
        assert False  # pragma: no cover
    except ParseException:
        pass

    try:
        parse('(;)W[tt]')  # a property has to be in a game
        assert False  # pragma: no cover
    except ParseException:
        pass

    try:
        parse('(;1)')  # property names can't start with numbers
        assert False  # pragma: no cover
    except ParseException:
        pass

    try:
        parse('(;FOO[bar]5)')  # bad character after a property value
        assert False  # pragma: no cover
    except ParseException:
        pass

    try:
        parse('(;')  # finished mid-gametree
        assert False  # pragma: no cover
    except ParseException:
        pass

    assert parse('(;)')[0].rest is None


if __name__ == '__main__':
    main()
