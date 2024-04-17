module add_data(clk,rst_n,data_in,data_out);

	input clk;
	input rst_n;
	input signed [127:0] data_in;
	output reg signed [20:0] data_out;
	
	always @(posedge clk or negedge rst_n) begin
		if(~rst_n) begin
			data_out<=0;
		end
		else begin
			data_out <= data_in[15:0] + data_in[31:16] + data_in[47:32] + 
			data_in[63:48] + data_in[79:64] + data_in[95:80] + 
			data_in[111:96] + data_in[127:112];
		end
	end
endmodule
	
	
	
	
	
	
	
	
	