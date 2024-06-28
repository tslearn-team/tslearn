"""Utility functions for doc building."""

from sphinx_gallery.scrapers import matplotlib_scraper

class MatplotlibSVGScraper(object):

    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return matplotlib_scraper(*args, format='svg', **kwargs)

matplotlib_svg_scraper = MatplotlibSVGScraper()
