build:
    mojo build src/main.mojo -o build/mojollama

run model="qwen3.5-4b-Q4_K_M.gguf":
    mojo run src/main.mojo -- --model {{model}}

fmt:
    mojo format src/

clean:
    rm -rf build/ *.mojopkg
