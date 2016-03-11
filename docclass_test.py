import unittest
import docclass

TEST_DATABASE = 'test.db'


class DocclassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = docclass.naivebayes(docclass.getwords)
        cls.db = cls.client.setdb(TEST_DATABASE)

    def setUp(self):
        docclass.sampletrain(self.client)

    def test_fprob(self):
        fpb = self.client.fprob('quick', 'good')
        self.assertGreaterEqual(fpb, 0.0)

    def test_prob(self):
        pb = self.client.prob('quick rabbit', 'good')
        self.assertGreaterEqual(pb, 0.0)

if __name__ == '__main__':
    unittest.main()
