function(genCompilation FILE_ITEM)
                get_filename_component(FILE_ITEM_WE ${FL_ITEM} NAME_WE)  

                set(EXTENSION "cpp")

                if(FL_ITEM MATCHES "cu.in$")
                     set(EXTENSION "cu") 
                endif()

                file(READ ${FL_ITEM} CONTENT_FL)
                #check content for types

                #set all to false 
                set (FLOAT_TYPE_GEN     0) 
                set (INT_TYPE_GEN       0) 
                set (LIBND4J_TYPE_GEN   0) 
                set (PAIRWISE_TYPE_GEN  0)
                set (RANGE_STOP        -1)

                string(REGEX MATCHALL "#cmakedefine[ \t]+[^_]+_TYPE_GEN" TYPE_MATCHES ${CONTENT_FL})
                #message("+++++++++++++++++++++++++++++++")
                foreach(TYPEX ${TYPE_MATCHES})   
                    set(STOP -1)
                    if(TYPEX MATCHES "INT_TYPE_GEN$")
                       set (INT_TYPE_GEN  1)
                       set(STOP 7)
                    endif()
                    if(TYPEX MATCHES "LIBND4J_TYPE_GEN$")
                       set (LIBND4J_TYPE_GEN 1)
                       set(STOP 9)
                    endif()
                    if(TYPEX MATCHES "FLOAT_TYPE_GEN$")
                       set (FLOAT_TYPE_GEN 1)
                       set(STOP 3)
                    endif()
                    if(TYPEX MATCHES "PAIRWISE_TYPE_GEN$")
                       set (PAIRWISE_TYPE_GEN  1)
                       set(STOP 12)
                    endif()
                    if(STOP GREATER RANGE_STOP) 
                       set(RANGE_STOP ${STOP})
                    endif()
                     
                endforeach()  

                foreach(FL_TYPE_INDEX RANGE 0 ${RANGE_STOP}) 
                    # set OFF if the index is above
                    if(FL_TYPE_INDEX GREATER 3)
                         set (FLOAT_TYPE_GEN     0) 
                    endif()
                    if(FL_TYPE_INDEX GREATER 7)
                         set (INT_TYPE_GEN     0) 
                    endif()
                    if(FL_TYPE_INDEX GREATER 9)
                         set (LIBND4J_TYPE_GEN   0) 
                    endif()                    
                    set(GENERATED_SOURCE  "${CMAKE_BINARY_DIR}/compilation_units/${FILE_ITEM_WE}_${FL_TYPE_INDEX}.${EXTENSION}") 
                    configure_file(  "${FL_ITEM}" "${GENERATED_SOURCE}" @ONLY)
                    LIST(APPEND CUSTOMOPS_GENERIC_SOURCES ${GENERATED_SOURCE} )
                endforeach()  
 

endfunction()