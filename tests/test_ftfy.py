import unittest

from smashed.mappers.text import FtfyMapper


class TestFtfyMapper(unittest.TestCase):
    def test_ftfy_mapper(self):

        # These are the test cases from the ftfy documentation
        dataset = [
            {"text": "âœ” No problems"},
            {"text": "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."},
            {"text": "l’humanitÃ©"},
            {"text": "Ã\xa0 perturber la rÃ©flexion"},
            {"text": "Ã perturber la rÃ©flexion"},
            {"text": "P&EACUTE;REZ"},
        ]

        mapper = FtfyMapper(input_fields="text")

        result = mapper.map(dataset)

        self.assertEqual(result[0], {"text": "✔ No problems"})
        self.assertEqual(
            result[1], {"text": "The Mona Lisa doesn't have eyebrows."}
        )
        self.assertEqual(result[2], {"text": "l'humanité"})
        self.assertEqual(result[3], {"text": "à perturber la réflexion"})
        self.assertEqual(result[4], {"text": "à perturber la réflexion"})
        self.assertEqual(result[5], {"text": "PÉREZ"})
