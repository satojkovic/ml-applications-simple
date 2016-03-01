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

if __name__ == '__main__':
    main()
