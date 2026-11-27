include(cc_test)

target_link_libraries(rwkv7_decoder_layer_test
                      PRIVATE
                      brpc
                      "$<LINK_GROUP:RESCAN,layers,common_layers>")
