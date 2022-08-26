mkdir -p ~/.streamlit/

echo "\
[server]\n\r\
headless = true\n\r\
enableCORS=true\n\r\
port = $PORT\n\r\
" > ~/.streamlit/config.toml
