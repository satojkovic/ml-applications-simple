#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docclass


def main():
    cl = docclass.classifier(docclass.getwords)
    docclass.sampletrain(cl)
    print cl.fprob('quick', 'good')
    print cl.weighted_prob('money', 'good', cl.fprob)
    docclass.sampletrain(cl)
    print cl.weighted_prob('money', 'good', cl.fprob)

    clnb = docclass.naivebayes(docclass.getwords)
    docclass.sampletrain(clnb)
    print clnb.prob('quick rabbit', 'good')
    print clnb.prob('quick rabbit', 'bad')
    print clnb.classify('quick rabbit', default='unknown')
    print clnb.classify('quick money', default='unknown')
    clnb.setthreshold('bad', 3.0)
    print clnb.classify('quick money', default='unknown')

    clfs = docclass.fisherclassifier(docclass.getwords)
    docclass.sampletrain(clfs)
    print clfs.cprob('quick', 'good')
    print clfs.cprob('money', 'bad')
    print clfs.weighted_prob('money', 'bad', clfs.cprob)
    print clfs.fisherprob('quick rabbit', 'good')
    print clfs.fisherprob('quick rabbit', 'bad')
    print clfs.classify('quick rabbit')
    print clfs.classify('quick money')

if __name__ == '__main__':
    main()
