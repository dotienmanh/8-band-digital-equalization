module slider_gain(clk,rst_n,data_fir_in,gain,data_gain_out);

	input clk;
	input rst_n;
	input signed [15:0] data_fir_in;
	input signed [4:0] gain;
	output reg signed [20:0] data_gain_out;
	
	
	always @(posedge clk or negedge rst_n) begin
		if(~rst_n) begin
			data_gain_out <= 16'b0;
		end
		else begin
			data_gain_out <= data_fir_in * gain;
		end
	end
	
endmodule