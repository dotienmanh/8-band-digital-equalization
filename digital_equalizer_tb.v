module Digital_Equalizer_tb();

	reg clk;
	reg clk_for_fir;
	reg rst_n;
	reg fir_ready;
	reg signed [15:0] in;
	reg signed [39:0] gain;
	wire signed [15:0] out;
	
	Digital_Equalizer uut(
		clk,
		clk_for_fir,
		rst_n,
		fir_ready,
		in,
		gain,
		out,
	);
	
	localparam max_addr = 100000;

	initial begin
		gain[4:0] = 5'b00001;
		gain[9:5] = 5'b00001;
		gain[14:10] = 5'b00000;
		gain[19:15] = 5'b00000;
		gain[24:20] = 5'b00000;
		gain[29:25] = 5'b00000;
		gain[34:30] = 5'b00000;
		gain[39:35] = 5'b00000;
		
	end
	
	initial begin
		clk=1;
		fir_ready=1;
		rst_n=1;
		forever #31250 clk = !clk;
	end

	initial begin
		clk_for_fir=1;
		forever #6250 clk_for_fir=!clk_for_fir;
	end
	
	integer outfile;
	reg signed [15:0] data_in[0:max_addr];
	integer i;
	
	initial begin
		$readmemb("input_for_tb.txt", data_in);
		outfile = $fopen("output_base.txt","w");
		for(i=0;i<max_addr;i=i+1) begin
			#62500
			in = data_in[i];
			$fdisplay(outfile,"%b",out);
		end
		$fclose(outfile);
	end
endmodule
	
	
	
	
	
