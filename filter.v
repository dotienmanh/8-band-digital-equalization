module fir_v2(
    clk,
    rst_n,
    fir_ready,
    data_in,
    tap,
    data_out
);

    parameter k = 15;

    input clk;
    input rst_n;
    input fir_ready;
    input signed [15:0] data_in;
    input signed [239:0] tap;
    output reg signed [31:0] data_out;

    reg signed [15:0] in_buff;
    reg enable_buff, enable_fir;
    reg signed [15:0] buff [14:0];
    reg signed [15:0] mul1, mul2 ;
    reg signed [31:0] out_buff;
    reg signed [31:0] mac_buff, mac_pre;

    reg [3:0] cnt;
    integer i;

    initial begin
        cnt = 3'b000;
    end

    always @(posedge clk) begin
        if(cnt == 4'he)begin
            cnt <= 3'b000;
            enable_buff <= 1'b1;
            enable_fir <= 1'b1;
        end else begin
            cnt <= cnt + 1'b1;
            enable_buff <= 1'b0;
            enable_fir <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if(~rst_n)begin
            in_buff <= 16'h0;
        end else if (fir_ready == 1'b1)begin
            in_buff <= data_in;
        end else begin
            in_buff <= 16'h0;
        end
    end

    always @(posedge clk ) begin
        if(enable_buff == 1'b1) begin
            buff [0] <= in_buff;
            for(i=1;i<k;i=i+1)begin
                buff[i] <= buff[i-1];
            end
        end else begin
            for (i = 0;i<k;i=i+1)begin
                buff[i] <= buff[i];
            end
        end
    end

    //Mux
    always @(cnt) begin
        if(cnt == 4'h0)begin
            mul1 = tap[239:224];
            mul2 = buff[0];
        end else if (cnt == 3'h1) begin
            mul1 = tap[223:208];
            mul2 = buff[1];
        end else if (cnt == 3'h2) begin
            mul1 = tap[207:192];
            mul2 = buff[2];
        end else if (cnt == 3'h3) begin
            mul1 = tap[191:176];
            mul2 = buff[3];
        end else if (cnt == 3'h4) begin
            mul1 = tap[175:160];
            mul2 = buff[4];
        end else if (cnt == 3'h5) begin
            mul1 = tap[159:144];
            mul2 = buff[5];
        end else if (cnt == 3'h6) begin
            mul1 = tap[143:128];
            mul2 = buff[6];
        end else if (cnt == 3'h7) begin
            mul1 = tap[127:112];
            mul2 = buff[7];
        end else if (cnt == 3'h8) begin
            mul1 = tap[111:96];
            mul2 = buff[8];
        end else if (cnt == 3'h9) begin
            mul1 = tap[95:80];
            mul2 = buff[9];
        end else if (cnt == 3'h10) begin
            mul1 = tap[79:64];
            mul2 = buff[10];
        end else if (cnt == 3'h11) begin
            mul1 = tap[63:48];
            mul2 = buff[11];
        end else if (cnt == 3'h12) begin
            mul1 = tap[47:32];
            mul2 = buff[12];
        end else if (cnt == 3'h13) begin
            mul1 = tap[31:16];
            mul2 = buff[13];
        end else if (cnt == 3'h14) begin
            mul1 = tap[15:0];
            mul2 = buff[14];
        end else begin
            mul1 = 0;
            mul2 = 0;
        end
        mac_buff = mul1*mul2 + out_buff;
    end

    //MAC
    always @(posedge clk) begin
        if(cnt == 3'b000)begin
            out_buff <= 0;
        end else begin
            out_buff <= mac_buff;
        end
    end


    //Output
    always @(posedge clk) begin
        if(enable_fir == 1'b1)begin
            data_out <= out_buff;
        end else begin
            data_out <= data_out;
        end
    end
    
endmodule