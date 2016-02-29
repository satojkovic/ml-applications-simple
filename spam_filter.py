#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docclass


def main():
    cl = docclass.classifier(docclass.getwords)
    cl.train('the quick brown fox jumps over the lazy dog', 'good')
    cl.train('make quick money in the online casino', 'bad')
    print cl.fcount('quick', 'good')


if __name__ == '__main__':
    main()
