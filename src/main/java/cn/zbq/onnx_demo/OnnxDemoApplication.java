package cn.zbq.onnx_demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class OnnxDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(OnnxDemoApplication.class, args);
        OnnxTest.main(args);
    }

}
