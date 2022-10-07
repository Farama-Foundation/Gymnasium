import os
from sphinx_gallery import gen_rst
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF

if __name__ == "__main__":
    source_path = os.path.join(os.path.dirname(__file__), "..", "_tutorials")
    dest_path = os.path.join(os.path.dirname(__file__), "..", "tutorials")
    for filename in os.listdir(source_path):
        if os.path.splitext(filename)[1] != ".py":
            continue

        file_path = os.path.join(source_path, filename)
        gallery_config = DEFAULT_GALLERY_CONF
        gallery_config["lang"] = "python"
        gallery_config["src_dir"] = file_path
        gallery_config["titles"] = {}
        gallery_config["titles"][file_path] = ""
        gallery_config["memory_base"] = 0.0
        gallery_config["min_reported_time"] = float("inf")
        gallery_config["exclude_implicit_doc_regex"] = True

        md_content = gen_rst.generate_file_rst(filename, dest_path, source_path, gallery_config)