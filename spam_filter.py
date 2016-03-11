#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docclass
import feedparser
import re


def feedclassifier(feed, classifier):
    f = feedparser.parse(feed)
    for entry in f['entries']:
        print
        print '-----'
        print 'Title', entry['title'].encode('utf-8')

        fulltext = '%s\n%s' % (entry['title'],
                               entry['content'])

        print 'Guess:', str(classifier.classify(fulltext))
        #cl = raw_input('Enter category: ')
        cl = 'python'
        classifier.train(fulltext, cl)


def main():
    cl = docclass.classifier(docclass.getwords)
    cl.setdb('test1.db')
    docclass.sampletrain(cl)
    print cl.fprob('quick', 'good')
    print cl.weighted_prob('money', 'good', cl.fprob)
    docclass.sampletrain(cl)
    print cl.weighted_prob('money', 'good', cl.fprob)

    clnb = docclass.naivebayes(docclass.getwords)
    clnb.setdb('test1.db')
    docclass.sampletrain(clnb)
    print clnb.prob('quick rabbit', 'good')
    print clnb.prob('quick rabbit', 'bad')
    print clnb.classify('quick rabbit', default='unknown')
    print clnb.classify('quick money', default='unknown')
    clnb.setthreshold('bad', 3.0)
    print clnb.classify('quick money', default='unknown')

    clfs = docclass.fisherclassifier(docclass.getwords)
    clfs.setdb('test1.db')
    docclass.sampletrain(clfs)
    print clfs.cprob('quick', 'good')
    print clfs.cprob('money', 'bad')
    print clfs.weighted_prob('money', 'bad', clfs.cprob)
    print clfs.fisherprob('quick rabbit', 'good')
    print clfs.fisherprob('quick rabbit', 'bad')
    print clfs.classify('quick rabbit')
    print clfs.classify('quick money')

    clfs2 = docclass.fisherclassifier(docclass.getwords)
    clfs2.setdb('test1.db')
    feedclassifier('feed_sample2.rss', clfs2)
    print clfs2.cprob('Pandas', 'python')
    print clfs2.cprob('python', 'python')

if __name__ == '__main__':
    main()
