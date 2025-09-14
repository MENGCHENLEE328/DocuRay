"""Router tests for intent routing and keywords."""  # Author: Team DocuRay | Generated: TDD step | Version: 0.1.0 | Modified: 2025-09-14

import unittest


class TestQueryRouter(unittest.TestCase):
    """Ensure router maps queries to expected routes."""

    def setUp(self):  # Arrange
        from docray.core.router import QueryRouter  # type: ignore
        self.router = QueryRouter()
        self.router.load_models()

    def test_exact_file_intent(self):  # Act/Assert
        out = self.router.analyze_query("查找文件名: report.pdf")
        self.assertEqual(out["route"], "fast_file_search")
        self.assertIn("intent", out)

    def test_code_intent(self):
        out = self.router.analyze_query("这个函数 handleError 在哪里？")
        self.assertEqual(out["route"], "ast_analysis_path")

    def test_table_intent(self):
        out = self.router.analyze_query("表格中的营收数据")
        self.assertEqual(out["route"], "table_extraction_path")

    def test_semantic_default(self):
        out = self.router.analyze_query("公司文化介绍")
        self.assertEqual(out["route"], "vector_search_path")


if __name__ == "__main__":
    unittest.main()

