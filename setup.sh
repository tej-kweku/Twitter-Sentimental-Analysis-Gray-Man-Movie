mkdir -p ~/.streamlit/

echo "\
[server]\n\r\
headless = true\n\r\
enableCORS=false\n\r\
port = $PORT\n\r\
" > ~/.streamlit/config.toml
