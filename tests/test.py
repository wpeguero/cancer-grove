"""Set of tests for observing the functionality of the algoriths."""
import unittest
import src.pipeline as pipeline

sample = "./data/CMMD-set/sample_data/D1-0820_1-4.dcm"

class TestPipeline(unittest.TestCase):
    """Test the data pipeline.

    Class made to test whether the pipeline is able to
    appropriately load data for training or testing. This
    will also be used to examine whether the data is
    extracted and/or transformed without encountering an
    unknown error.
    """
    def test_data_extraction(self):
        self.assertEqual(type(pipeline.extract_data(sample)), dict)
    
    def test_data_transformation(self):
        datapoint = pipeline.extract_data(sample)
        datapoint = pipeline.transform_data(datapoint)
        for key, value in datapoint.items():
            if (key == 'Subject ID') or (key == 'image'):
                pass
            else: 
                self.assertEqual(type(value), int)

    def test_rescale_imaging(self):
        pass


if __name__ == "__main__":
    unittest.main()
