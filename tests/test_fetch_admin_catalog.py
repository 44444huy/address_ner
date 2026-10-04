import unittest

from address_ner.fetch_admin_catalog import (
    build_page_callback,
    normalize_rows,
    parse_form_inputs,
    parse_grid_rows,
)


class AdminCatalogImportTests(unittest.TestCase):
    def test_parses_form_inputs_needed_by_callback(self):
        html = """
        <input type="hidden" name="__VIEWSTATE" value="state">
        <input type="text" name="date" value="01/07/2025">
        <input type="submit" name="download" value="Excel">
        """

        self.assertEqual(
            parse_form_inputs(html),
            {"__VIEWSTATE": "state", "date": "01/07/2025"},
        )

    def test_builds_zero_based_page_callback(self):
        self.assertEqual(
            build_page_callback(1),
            "GB|20;12|PAGERONCLICK3|PN1;",
        )

    def test_removes_javascript_escaped_whitespace_from_cells(self):
        html = (
            '<tr id="ctl00_PlaceHolderMain_gridXa_DXDataRow0">'
            '<td>\\r\\n\\tXã Toàn Thắng</td></tr>'
        )

        self.assertEqual(parse_grid_rows(html), [["Xã Toàn Thắng"]])

    def test_parses_rows_from_a_selected_grid(self):
        html = (
            '<tr id="ctl00_PlaceHolderMain_gridTinh_DXDataRow0">'
            '<td>01</td><td>Thành phố Hà Nội</td></tr>'
        )

        self.assertEqual(
            parse_grid_rows(html, grid_name="gridTinh"),
            [["01", "Thành phố Hà Nội"]],
        )

    def test_parses_and_normalizes_current_unit_columns(self):
        cells = [
            "01",
            "Thành phố Hà Nội",
            "00004",
            "Phường Ba Đình",
            "decision-old",
            "01/07/2025",
            "Phường Ba Đình",
            "00004",
            "decision-current",
            "01/07/2025",
            "Thành phố Hà Nội",
            "01",
            "note",
        ]
        row_html = (
            '<tr id="ctl00_PlaceHolderMain_gridXa_DXDataRow0">'
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )

        rows = parse_grid_rows(row_html)
        units = normalize_rows(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(units[0]["province_code"], "01")
        self.assertEqual(units[0]["ward_code"], "00004")
        self.assertEqual(units[0]["unit_type"], "PHƯỜNG")
        self.assertEqual(units[0]["short_name"], "Ba Đình")

    def test_normalizes_decomposed_vietnamese_characters(self):
        cells = [
            "01",
            "Thành phố Hà Nội",
            "00001",
            "Xã Trung Giã",
            "decision-old",
            "01/07/2025",
            "Xã Trung Giã",
            "00001",
            "decision-current",
            "01/07/2025",
            "Thành phố Hà Nội",
            "01",
            "note",
        ]

        units = normalize_rows([cells])

        self.assertEqual(units[0]["ward_name"], "Xã Trung Giã")
        self.assertEqual(units[0]["unit_type"], "XÃ")


if __name__ == "__main__":
    unittest.main()
