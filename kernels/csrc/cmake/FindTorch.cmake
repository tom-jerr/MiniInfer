# FindTorch.cmake - Helper module to find PyTorch
# This is a fallback if the standard find_package doesn't work

if(NOT Torch_FOUND)
    # Try to find torch via Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
        OUTPUT_VARIABLE TORCH_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_RESULT
    )
    
    if(TORCH_RESULT EQUAL 0)
        set(TORCH_INCLUDE_DIRS 
            "${TORCH_PATH}/include"
            "${TORCH_PATH}/include/torch/csrc/api/include"
        )
        
        # Find torch libraries
        find_library(TORCH_LIBRARY torch PATHS "${TORCH_PATH}/lib" NO_DEFAULT_PATH)
        find_library(C10_LIBRARY c10 PATHS "${TORCH_PATH}/lib" NO_DEFAULT_PATH)
        find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_PATH}/lib" NO_DEFAULT_PATH)
        find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_PATH}/lib" NO_DEFAULT_PATH)
        
        set(TORCH_LIBRARIES 
            ${TORCH_LIBRARY}
            ${C10_LIBRARY}
            ${TORCH_CPU_LIBRARY}
        )
        
        if(TORCH_CUDA_LIBRARY)
            list(APPEND TORCH_LIBRARIES ${TORCH_CUDA_LIBRARY})
        endif()
        
        set(Torch_FOUND TRUE)
    endif()
endif()
