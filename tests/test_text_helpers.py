import unittest

import torch

import text_helpers as th

class TestUnicodeToAsciiFunction(unittest.TestCase):
    '''Test unicodeToAsciiFunction function'''

    def test_certain_special_characters_conversion(self):
        '''Test for correct conversion of special characters by the function'''
        characters = [
            ('À', 'A'),
            ('à', 'a'),
            ('ã', 'a'),
            ('Ć', 'C')
        ]
        for special_character, expected_character in characters:
            with self.subTest(special_char=special_character, expected_char=expected_character):
                self.assertEqual(th.unicodeToAscii(special_character), expected_character)


class TestGetInputWordTensorFunction(unittest.TestCase):
    '''Test getInputWordTensor function'''

    def test_for_one_character_sequence(self):
        '''Test for "a" sequence'''
        assert th.getInputWordTensor('a').shape == (1, 1, th.LETTERS_TOTAL)
        t = torch.zeros(1, 1, th.LETTERS_TOTAL)
        t[0][0][0] = 1
        assert torch.all(torch.eq(t, th.getInputWordTensor('a')))

    def test_for_two_character_sequence(self):
        '''Test for "a'" sequence'''
        assert th.getInputWordTensor("a'").shape == (2, 1, th.LETTERS_TOTAL)
        t = torch.zeros(2, 1, th.LETTERS_TOTAL)
        t[0][0][0] = 1
        t[1][0][th.LETTERS_TOTAL - 2] = 1 # -2 coz the -1 is EOS
        assert torch.all(torch.eq(t, th.getInputWordTensor("a'")))

    def test_for_three_character_sequence(self):
        '''Test for "az'" sequence'''
        assert th.getInputWordTensor("az'").shape == (3, 1, th.LETTERS_TOTAL)
        t = torch.zeros(3, 1, th.LETTERS_TOTAL)
        t[0][0][0] = 1
        t[1][0][25] = 1 # Index for letter z
        t[2][0][th.LETTERS_TOTAL - 2] = 1 # -2 coz the -1 is EOS
        assert torch.all(torch.eq(t, th.getInputWordTensor("az'")))


class TestGetTargetWordTensor(unittest.TestCase):
    '''Test getTargetWordTensor function'''

    def test_for_one_character_sequence(self):
        '''Test for "a" sequence'''
        assert list(th.getTargetWordTensor('a').shape) == [2]
        assert torch.all(torch.eq(th.getTargetWordTensor('a'), torch.LongTensor([0, 53])))

    def test_for_two_character_sequence(self):
        '''Test for "a'" sequence'''
        assert list(th.getTargetWordTensor("a'").shape) == [3]
        assert torch.all(torch.eq(th.getTargetWordTensor("a'"), torch.LongTensor([0, 52, 53])))

    def test_for_three_character_sequence(self):
        '''Test for "az'" sequence'''
        assert list(th.getTargetWordTensor("az'").shape) == [4]
        assert torch.all(torch.eq(th.getTargetWordTensor("az'"), torch.LongTensor([0, 25, 52, 53])))
