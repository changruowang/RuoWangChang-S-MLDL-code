# --*-- coding:utf-8 --*--
import sys
import pygraphviz as pgv

'''
treeWriter with func wite can write a binary tree to tree.png or user spcified
file
'''

class treeWriter():
    def __init__(self, tree):
        self.num = 1  # mark each visible node as its key
        self.num2 = -1  # makk each invisible node as its key
        self.tree = tree
        self.A = None
    def write(self, outfile='tree.png'):
        def writeHelp(root, A):
            if not root:
                return

            p = str(self.num)
            self.num += 1
            A.add_node(p, label=str(root.date))
            q = None
            r = None

            if root.left:
                q = writeHelp(root.left, A)
                A.add_node(q, label=str(root.left.date))
                A.add_edge(p, q)
            if root.right:
                r = writeHelp(root.right, A)
                A.add_node(r, label=str(root.right.date))
                A.add_edge(p, r)

            if q or r:
                if not q:
                    q = str(self.num2)
                    self.num2 -= 1
                    A.add_node(q, style='invis')
                    A.add_edge(p, q, style='invis')
                if not r:
                    r = str(self.num2)
                    self.num2 -= 1
                    A.add_node(r, style='invis')
                    A.add_edge(p, r, style='invis')
                l = str(self.num2)
                self.num2 -= 1
                A.add_node(l, style='invis')
                A.add_edge(p, l, style='invis')
                B = A.add_subgraph([q, l, r], rank='same')
                B.add_edge(q, l, style='invis')
                B.add_edge(l, r, style='invis')

            return p  # return key root node

        self.A = pgv.AGraph(directed=True, strict=True)
        writeHelp(self.tree.KdTree, self.A)
        self.A.graph_attr['epsilon'] = '0.001'
        # print self.A.string()  # print dot file to standard output
        self.A.layout('dot')  # layout with dot
        self.A.draw(outfile)  # write to file


