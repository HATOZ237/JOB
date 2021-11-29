----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 01.10.2021 14:21:27
-- Design Name: 
-- Module Name: Reg_8bits - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

library Logic_com;
use Logic_com.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity Reg_8bits is
    Port ( reset : in STD_LOGIC;
           datain : in STD_LOGIC;
           dataout : out std_logic_vector(7 downto 0);
           clk : in STD_LOGIC;
           enable : in STD_LOGIC);
end Reg_8bits;

architecture Behavioral of Reg_8bits is
signal sin: std_logic_vector(7 downto 0) := (others => '0');
signal sout: std_logic_vector(7 downto 0) := (others => '0');


begin
reg_0 : entity Logic_com.registre_1 
port map(rst => reset, en=> enable, clk=>clk, d=> sin(0), q=>sout(0));

reg_1 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(1), q=>sout(1));

reg_2 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(2), q=>sout(2));

reg_3 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(3), q=>sout(3));

reg_4 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(4), q=>sout(4));

reg_5 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(5), q=>sout(5));

reg_6 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(6), q=>sout(6));

reg_7 : entity Logic_com.registre_1
port map(rst => reset, en=> enable, clk=>clk, d=> sin(7), q=>sout(7));

sin(0) <= datain;

sin(1) <= sout(0);
sin(2) <= sout(1);
sin(3) <= sout(2);
sin(4) <= sout(3);
sin(5) <= sout(4);
sin(6) <= sout(5);
sin(7) <= sout(6);

dataout <= sout;

end Behavioral;
